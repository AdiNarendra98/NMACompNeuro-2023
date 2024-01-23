#Launcher function to rn run_full_analysis with swiped parameters

import os

feature = "cursorX"
p_ind = 1
window_size = 10000
bn = 6
f1 = 8
f2 = 25
num_points = 300
forc_dist = 0

# Search window size
# for window_size in np.linspace(100,500,50).astype(np.int32):
for p_ind in range(4):
    for feature in ["cursorY", "targetX", "targetY", "cursor_angle", "cursor_radius", "target_angle", "target_radius"]:
        # for window_size in range(4000,14000,2000):
        # for f1,f2 in list(zip([8,12,35], [12,24,42])):
        for f1,f2 in list(zip([12],[24])):
            for bn in [4,6,38]:
                os.system(f"python run_full_analysis.py --feature {feature} --window_size {window_size} --num_points {num_points} --brodmann_area_number {bn} --frequency_band {f1} {f2} --participant_ind {p_ind} --forc_dist {forc_dist}")