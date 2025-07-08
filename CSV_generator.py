import os
import csv
import re

root_dir = 'D:\Milling prediction\Milling Surface Roughness Acoustic Sensor Dataset'  # 请修改为你的路径

csv_path = 'audio_labels.csv'

def extract_param(folder_name, pattern):
    match = re.search(pattern, folder_name)
    if match:
        return match.group(1)
    return None

with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Ra','feed_rate_mm', 'rpm', 'depth_mm', 'filepath'])
    
    # 遍历feed-开头的目录
    for feed_folder in os.listdir(root_dir):
        if not feed_folder.startswith('feed-'):
            continue
        feed = extract_param(feed_folder, r'feed-(\d+)mm')
        feed_path = os.path.join(root_dir, feed_folder)
        for rpm_folder in os.listdir(feed_path):
            if not rpm_folder.startswith('speed-'):
                continue
            rpm = extract_param(rpm_folder, r'speed-(\d+)rpm')
            rpm_path = os.path.join(feed_path, rpm_folder)
            for doc_folder in os.listdir(rpm_path):
                if not doc_folder.startswith('doc-'):
                    continue
                depth = extract_param(doc_folder, r'doc-(\d+\.\d+)')
                ra_value = extract_param(doc_folder, r'Ra-(\d+\.\d+)')
                doc_path = os.path.join(rpm_path, doc_folder)
                for file in os.listdir(doc_path):
                    datanumber=os.path.join(doc_path, file)
                    for data in os.listdir(datanumber):
                        if not data.endswith('_data'):
                            continue
                        inner_folder= os.path.join(datanumber, data)
                        data_path= os.path.join(inner_folder, 'e00', 'd00')
                        for datas in os.listdir(data_path):
                            if datas.endswith('.au'):
                                filepath = os.path.join(data_path, datas)
                                writer.writerow([ra_value, feed, rpm, depth, filepath])