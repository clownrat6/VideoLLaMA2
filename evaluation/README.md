# Evaluation Data Structure

```
VideoLLaMA2
├── benchs
│   ├── egoschema # Official website: https://github.com/egoschema/EgoSchema
|   |   ├── good_clips_git/ # Available at: https://drive.google.com/drive/folders/1SS0VVz8rML1e5gWq7D7VtP1oxE2UtmhQ
|   |   └── questions.json  # Available at: https://github.com/egoschema/EgoSchema/blob/main/questions.json
│   ├── mvbench # Official website: https://huggingface.co/datasets/OpenGVLab/MVBench
|   |   ├── video/ # move videos in data_0613/ to correct folder, download nturgbd.zip from https://huggingface.co/datasets/PKU-Alignment/MVBench/resolve/main/video/nturgbd.zip
|   |   |   ├── clever/
|   |   |   └── ...
|   |   └── json/
|   |   |   ├── action_antonym.json
|   |   |   └── ...
│   ├── perception_test_mcqa # Official website: https://github.com/google-deepmind/perception_test
|   |   ├── videos/ # Available at: https://storage.googleapis.com/dm-perception-test/zip_data/test_videos.zip
|   |   └── mc_question_test.json # Download from https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_test_annotations.zip
│   ├── videomme # Official website: https://video-mme.github.io/home_page.html#leaderboard
|   |   ├── test-00000-of-00001.parquet
|   |   ├── videos/
|   |   └── subtitles/
│   ├── activitynet_qa # Official website: https://github.com/MILVLG/activitynet-qa
|   |   ├── all_test/   # Available at: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW
|   |   ├── test_q.json # Available at: https://github.com/MILVLG/activitynet-qa/tree/master/dataset
|   |   └── test_a.json # Available at: https://github.com/MILVLG/activitynet-qa/tree/master/dataset
│   ├── vcgbench # Official website: https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/quantitative_evaluation
|   |   ├── Test_Videos/ # Available at: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW
|   |   ├── generic_qa.json     # These three json files available at: https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa
|   |   ├── temporal_qa.json
|   |   └── consistency_qa.json
│   ├── mlvu # Official website: https://github.com/JUNJIE99/MLVU
|   |   ├── video/ # download from this folder: https://huggingface.co/datasets/MLVU/MVLU/tree/main/MLVU
|   |   └── json/
│   ├── nextqa # Official website: https://github.com/doc-doc/NExT-QA
|   |   ├── NExTVideo/ # download from google drive: https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view
|   |   ├── train.csv  # These four files available at: https://github.com/doc-doc/NExT-QA/tree/main/dataset/nextqa
|   |   ├── val.csv
|   |   ├── test.csv
|   |   └── map_vid_vidorID.json
│   ├── tomato # Official website: https://github.com/yale-nlp/TOMATO
|   |   ├── data/ # download from here: https://github.com/yale-nlp/TOMATO/tree/main/data
|   |   └── videos/ # download from here: https://drive.google.com/file/d/1-dNt9bZcp6C3RXuGoAO3EBgWkAHg8NWR/view
│   ├── longvideobench # Official website: https://longvideobench.github.io/
|   |   ├── videos/ # download videos.tar.part.* from here: https://huggingface.co/datasets/longvideobench/LongVideoBench/tree/main
|   |   ├── subtitles/ # download from here: https://huggingface.co/datasets/longvideobench/LongVideoBench/tree/main
|   |   ├── lvb_test_wo_gt.json
|   |   └── lvb_val.json
│   ├── tempcompass # Official website: https://github.com/llyx97/TempCompass
|   |   ├── videos/ # all can be downloaded from here: https://huggingface.co/datasets/lmms-lab/TempCompass/tree/main
|   |   ├── captioning/test-00000-of-00001.parquet
|   |   ├── yes_no/test-00000-of-00001.parquet
|   |   ├── multi-choice/test-00000-of-00001.parquet
|   |   └── caption_matching/test-00000-of-00001.parquet
│   ├── videovista # Official website: https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master/VideoVista
|   |   ├── merged/ # download merged.zip.* from here and merge them for unzip: https://huggingface.co/datasets/Uni-MoE/VideoVista/tree/main
|   |   ├── images/ # download relation_images.zip from here: https://huggingface.co/datasets/Uni-MoE/VideoVista/tree/main
|   |   └── VideoVista.json
```
