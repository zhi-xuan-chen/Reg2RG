import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import torch.nn.functional as F
import tqdm
import random
import pickle

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]
    
class RadGenomeDataset_Region_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=1024, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # 读取 JSON 文件
        with open(self.wrong_path, 'r', encoding='utf-8') as file:
            wrong_cases = json.load(file)

        # 初始化一个空列表
        wrong_list = []

        # 遍历字典并合并所有值到列表中
        for key, value in wrong_cases.items():
            wrong_list.extend(value)

        print('Number of wrong files: ', len(wrong_list))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    if accession_number in wrong_list:
                        continue

                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue

                        # # load img and mask
                        # img_data = nib.load(nii_file).get_fdata()
                        # mask_data = nib.load(mask_file).get_fdata()
                        # # check if the shape of the mask is the same as the img
                        # if img_data.shape != mask_data.shape:
                        #     continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        samples.append((accession_number, nii_file, mask_file, region_report, region))

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_path, transform):
        img_data = nib.load(img_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        if img_data.shape != mask_data.shape:
            print('Shape mismatch: ', img_path, mask_path)
            print('img shape: ', img_data.shape)
            print('mask shape: ', mask_data.shape)
            exit()

        mask_img = img_data * mask_data

        mask_img[mask_data == 0] = -1024

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        mask_img = np.clip(mask_img, hu_min, hu_max)

        mask_img = (((mask_img+400 ) / 600)).astype(np.float32)
        mask_img = mask_img[np.newaxis, ...]
        tensor = transform(mask_img)

        # repeat the tensor to have 3 channels
        tensor = tensor.repeat(3, 1, 1, 1)

        tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

        return tensor

    def text_add_image_tokens(self, text):
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, mask_file, region_report, region = self.samples[index]
        mask_img_tensor = self.mask_img_to_tensor(img_file, mask_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to the {region} region of this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential.".format(region=region)
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'region': region, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': mask_img_tensor, 
                'gt_region_report': region_report
                }
    
class RadGenomeDataset_Combined_Region_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, max_region_size=10, max_img_size = 1, image_num = 32, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<region>", "</region>", "<image>", "</image>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(image_num):
                region_token = "<region"+str(i*image_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*image_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)
        
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, transform = self.transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, transform):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path).get_fdata()

            # NOTE: check whether the mask is empty
            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()

        return mask_img_tensors

    def text_add_image_tokens(self, text):
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + text
        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }
    
class RadGenomeDataset_Combined_Region_Image_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, max_region_size=10, max_img_size = 1, image_num = 32, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(image_num):
                region_token = "<region"+str(i*image_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*image_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, transform = self.transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, transform):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path).get_fdata()

            # NOTE: check whether the mask is empty
            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data[np.newaxis, ...]
        tensor = transform(img_data)
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+400 ) / 600)).float()
        
        # repeat the tensor to have 3 channels
        tensor = tensor.repeat(3, 1, 1, 1)

        tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = tensor

        return mask_img_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

# NOTE: implement the version for batch inference
class RadGenomeDataset_Combined_Region_Image_Test_Batch(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, max_region_size=10, max_img_size = 1, image_num = 32, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(image_num):
                region_token = "<region"+str(i*image_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*image_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, transform = self.transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        if os.path.exists('/home/chenzhixuan/Workspace/FineGrainedCTRG/src/Dataset/valid_samples.pkl'):
            samples = pickle.load(open('/home/chenzhixuan/Workspace/FineGrainedCTRG/src/Dataset/valid_samples.pkl', 'rb'))
        else:
            # # TODO: comment this line after debug
            # i = 0
            for patient_folder in tqdm.tqdm(patient_folders):
                accession_folders = glob.glob(os.path.join(patient_folder, '*'))

                for accession_folder in accession_folders:
                    nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                    for nii_file in nii_files:
                        accession_number = nii_file.split("/")[-1]

                        if accession_number not in self.accession_to_sentences:
                            continue

                        # # remove some samples with wrong shape
                        # if accession_number in wrong_list:
                        #     continue
                            
                        single_sample = {}
                        volume_name = accession_number.split(".")[0]
                        mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                        single_sample['accnum'] = accession_number
                        # add nii_file to single_sample
                        single_sample['image'] = nii_file

                        for region in REGIONS:
                            mask_file = os.path.join(mask_path, region + '.nii.gz')
                            # NOTE: if the mask file does not exist, skip this sample
                            if not os.path.exists(mask_file):
                                continue
                            
                            # NOTE: if the region is not in the report, set the region report to ''
                            if region in self.accession_to_sentences[accession_number]:
                                region_report = self.accession_to_sentences[accession_number][region]
                            else:
                                region_report = ''
                            
                            single_sample[region] = [mask_file, region_report]

                        samples.append(single_sample)
                        self.paths.append(nii_file)
                #         # TODO: comment this line after debug
                #         i = i + 1
                #         if i > 1:
                #             break
                #     if i > 1:
                #         break
                # if i > 1:
                #     break
                
            # save the samples to a file
            with open('/home/chenzhixuan/Workspace/FineGrainedCTRG/src/Dataset/valid_samples.pkl', 'wb') as f:
                pickle.dump(samples, f)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, transform):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path).get_fdata()

            # NOTE: check whether the mask is empty
            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data[np.newaxis, ...]
        tensor = transform(img_data)
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+400 ) / 600)).float()
        
        # repeat the tensor to have 3 channels
        tensor = tensor.repeat(3, 1, 1, 1)

        tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = tensor

        return mask_img_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # NOTE: do not shuffle the regions during testing, so that the regions order won't change with random seed
        # random.shuffle(shuffled_areas) 

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # # make text input
        # text_tensor = self.text_tokenizer(
        #     prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        # )
        # text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'prompt': prompt,
                'vision_x': mask_img_tensors, 
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_Combined_Region_Image_Mask_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }
        
class RadGenomeDataset_Combined_Region_Image_Mask_Whole_No_Region_Report_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file
                    
                    single_sample['whole'] = self.accession_to_sentences[accession_number]['whole']

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']
        
        whole_report = self.data[index]['whole']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum' or key == 'whole':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        # NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ". " 
        
        gt_report = combined_report + "The whole report is: " + whole_report 

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': gt_report
                }

class RadGenomeDataset_RF_RR_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, inferenced_id, max_region_size=10, region_num=32, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided regional information from this CT scan, please generate a comprehensive medical report for each region. "
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_RF_MF_RR_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, inferenced_id, max_region_size=10, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided regional information from this CT scan, please generate a comprehensive medical report for each region. "
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_RF_MF_TS_RR_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, inferenced_id, max_region_size=10, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_RF_MF_GF_RR_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_Undecoupled_Region_Image_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=32, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        mask_imgs = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            
            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue
            mask_keys.append(key)

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(mask_img, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            mask_imgs.append(tensor)

            flag = True

        if not flag:
            print('No mask: ', img_path)
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        mask_imgs_data = torch.stack(mask_imgs, axis=0)
        tensors = image_transform({'img': img_data, 'seg': mask_imgs_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        mask_imgs_tensor = tensors['seg']
        for i, key in enumerate(mask_keys):
            mask_img_tmp = mask_imgs_tensor[i].unsqueeze(0)
            mask_img_tmp = mask_img_tmp.repeat(3, 1, 1, 1)
            mask_img_tensors[key] = mask_img_tmp.unsqueeze(0)

        return mask_img_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole'))                        

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+400 ) / 600)).float()
        
        # repeat the tensor to have 3 channels
        tensor = tensor.repeat(3, 1, 1, 1)

        tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

        return tensor

    def text_add_image_tokens(self, text):
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }
        
class RadGenomeDataset_M3D_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": []}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole'))                        

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+1000 ) / 1200)).float()
        
        tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    def text_add_image_tokens(self, text):
        text =  self.image_padding_tokens[0] + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }

class RadGenomeDataset_M3D_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": []}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole'))                        

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+1000 ) / 1200)).float()
        
        tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    def text_add_image_tokens(self, text):
        text =  self.image_padding_tokens[0] + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }
        
class RadGenomeDataset_CT2Rep_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": []}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole'))                        

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+1000 ) / 1200)).float()
        
        tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    def text_add_image_tokens(self, text):
        text =  self.image_padding_tokens[0] + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }

class RadGenomeDataset_MedVInT_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, inferenced_id, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": []}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole')) 
                    if accession_number in self.inferenced_id:
                        samples.pop()                 

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+1000 ) / 1200)).float()
        
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.squeeze(0)

        return tensor

    def text_add_image_tokens(self, text):
        text =  self.image_padding_tokens[0] + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }

class RadGenomeDataset_R2GenGPT_Image_Test(Dataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, wrong_path, max_img_size = 1, image_num = 32, max_seq=2048, voc_size=32000, ):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        self.image_padding_tokens = []
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": []}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.wrong_path = wrong_path

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.img_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                    
                    region_report = self.accession_to_sentences[accession_number]['whole']
                    samples.append((accession_number, nii_file, region_report, 'whole'))                        

                    self.paths.append(nii_file)
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, img_path, transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        tensor = transform(img_data)

        hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
        tensor = torch.clamp(tensor, hu_min, hu_max)

        tensor = (((tensor+1000 ) / 1200)).float()
        
        tensor = tensor.permute(0, 3, 1, 2) # shape: (1, 64, 256, 256)
        tensor = tensor.squeeze(0)

        return tensor

    def text_add_image_tokens(self, text):
        text =  self.image_padding_tokens[0] + text
        return text


    def __getitem__(self, index):
        accession_number, img_file, region_report, _ = self.samples[index]
        img_tensor = self.img_to_tensor(img_file)

        instruction = "Based on the above visual information, please generate a complete medical report corresponding to this CT scan. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential."
        
        prompt = self.text_add_image_tokens(instruction)

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': accession_number, 
                'question': prompt, 
                'lang_x': text_input, 
                'vision_x': img_tensor, 
                'gt_report': region_report
                }
        
class RadGenomeDataset_Bert_Combined_Region_Image_Mask_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["[image]", "[/image]", "[region]", "[/region]"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "[image"+str(i*image_num+j)+"]"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "[image"+str(i*image_num+j)+"]")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "[region"+str(i*region_num+j)+"]"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "[region"+str(i*region_num+j)+"]")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }

class RadGenomeDataset_GPT2_Combined_Region_Image_Mask_Test(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, inferenced_id, max_region_size=10, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["[image]", "[/image]", "[region]", "[/region]"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "[image"+str(i*image_num+j)+"]"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "[image"+str(i*image_num+j)+"]")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "[region"+str(i*region_num+j)+"]"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "[region"+str(i*region_num+j)+"]")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # # 获取目录下的所有文件名
        # file_names = os.listdir(self.wrong_path)

        # # 将文件名整理到一个列表中
        # wrong_list = []
        # for file_name in file_names:
        #     file_name, _ = os.path.splitext(file_name)
        #     wrong_list.append(file_name)

        # print('Number of wrong files: ', len(wrong_list))

        # # TODO: comment this line after debug
        # i = 0
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue

                    # # remove some samples with wrong shape
                    # if accession_number in wrong_list:
                    #     continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
                        
            #         # TODO: comment this line after debug
            #         i = i + 1
            #         if i > 1:
            #             break
            #     if i > 1:
            #         break
            # if i > 1:
            #     break 
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path, mmap=True)
        img_data = np.asarray(img_data.dataobj)
        img_data = torch.from_numpy(img_data).float()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)
            # if mask_img.max() < 0:
            #     print(mask_img.max())
            #     print(mask_path)
            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        # random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."
        
        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        return {'acc_num': self.data[index]['accnum'],
                'lang_x': text_input, 
                'vision_x': mask_img_tensors, 
                'mask_x': mask_tensors,
                'region2area': region2area,
                'question': prompt,
                'gt_combined_report': combined_report
                }