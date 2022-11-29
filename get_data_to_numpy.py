import pandas as pd

def get_removed_index_from_landmarks(right_hand,left_hand):
    removed_indexes = []
    for i in range(len(right_hand)):
        if(len(right_hand) - len(removed_indexes) <=60):
            break
        if(right_hand[i] == None and left_hand[i] == None):
            removed_indexes.append(i)
    return removed_indexes

def get_removed_index_from_world_landmarks(right_hand,left_hand):
    removed_indexes = []
    # print(len(right_hand))
    for i in range(len(right_hand)):
        
        if(len(right_hand) - len(removed_indexes) <=60):
            break
        # print(i,right_hand[i][0] == None and left_hand[i][0] == None)
        if(right_hand[i][0] == None and left_hand[i][0] == None):
            removed_indexes.append(i)
    return removed_indexes

def convert_to_60_frames(data):
    for item in data.iterrows():
        # remove_indexes = get_removed_index_from_landmarks(item[1].RIGHT_HAND_LANDMARKS,item[1].LEFT_HAND_LANDMARKS)
        # print(item)
        world_remove_indexes = get_removed_index_from_world_landmarks(item[1].RIGHT_HAND_WORLD_LANDMARKS,item[1].LEFT_HAND_WORLD_LANDMARKS)

        if(len(item[1].RIGHT_HAND_WORLD_LANDMARKS)<60):
            current_length = len(item[1].RIGHT_HAND_WORLD_LANDMARKS)
            for i in range(current_length,60):
                item[1].RIGHT_HAND_WORLD_LANDMARKS.append([None])
                item[1].LEFT_HAND_WORLD_LANDMARKS.append([None])
                item[1].POSE_WORLD_LANDMARKS.append([None])

        
        for index in sorted(world_remove_indexes, reverse=True):
            del item[1].RIGHT_HAND_WORLD_LANDMARKS[index]
            del item[1].LEFT_HAND_WORLD_LANDMARKS[index]
            del item[1].POSE_WORLD_LANDMARKS[index]
            
        if(len(item[1].RIGHT_HAND_WORLD_LANDMARKS)>60):
            del item[1].RIGHT_HAND_WORLD_LANDMARKS[60:]
            del item[1].LEFT_HAND_WORLD_LANDMARKS[60:]
            del item[1].POSE_WORLD_LANDMARKS[60:]

            
#Loop throug data
sequences = []
labels = []
import numpy as np
def extract_keypoints(pose_world_landmarks,right_hand_world_landmarks,left_hand_world_landmarks):
    pose = np.array([[res['X'], res['Y']] for res in pose_world_landmarks]).flatten() if pose_world_landmarks[0] else np.zeros(12*2)
    lh = np.array([[res['X'], res['Y'], res['Z']] for res in left_hand_world_landmarks ]).flatten() if left_hand_world_landmarks[0] else np.zeros(21*3)
    rh = np.array([[res['X'], res['Y'], res['Z']] for res in right_hand_world_landmarks ]).flatten() if right_hand_world_landmarks[0] else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def convert_world_to_2d_array(data):
    k = 0;
    for item in data.iterrows():
        sequence = []
        labels.append(item[1].Label)
        for i in range(len(item[1].RIGHT_HAND_WORLD_LANDMARKS)):    
            pose_world = item[1].POSE_WORLD_LANDMARKS[i];
            if pose_world == None:
                pose_world = [None]
            points = extract_keypoints(pose_world,item[1].RIGHT_HAND_WORLD_LANDMARKS[i],item[1].LEFT_HAND_WORLD_LANDMARKS[i])
            sequence.append(points)
        sequences.append(sequence)
        k= k+1


first = pd.read_json('PROCESSED_VIDEO_DATA.json',orient='index')
second = pd.read_json('PROCESSED_VIDEO_DATA1.json',orient='index')
third = pd.read_json('PROCESSED_VIDEO_DATA2.json',orient='index')

fourth = pd.read_json('PROCESSED_VIDEO_DATA3.json',orient='index')

print("OPENED FILES")

convert_to_60_frames(first)
convert_to_60_frames(second)
convert_to_60_frames(third)
convert_to_60_frames(fourth)
print("CONVERTED TO 60 FRAMES")
convert_world_to_2d_array(first)
convert_world_to_2d_array(second)
convert_world_to_2d_array(fourth)
convert_world_to_2d_array(third)

print("CONVERTED TO ARRAY")
X = np.array(sequences)
words = np.array(labels).reshape(-1,1)

with open('DATA.npy', 'wb') as f:
    np.save(f, X)
    
with open('LABELS.npy', 'wb') as f:
    np.save(f, words)