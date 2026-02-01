lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=kanishk_lerobot_leader_2 --display_data=true --dataset.repo_id=roborage/record_vtest --dataset.num_episodes=10 --dataset.single_task="Plug charger into the socket"

lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --dataset.repo_id=roborage/record_v14 --dataset.episode=0 

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=roborage/eval_groot-15k_1 --dataset.num_episodes=1 --dataset.single_task="Plug charger into the socket" --policy.path=/home/schada/.cache/huggingface/hub/models--igkp--groot-15k-small/snapshots/2f82606eaeffc30d26d93416b1c5b61440b3b769 --dataset.episode_time_s=30  --dataset.reset_time_s=10 --dataset.push_to_hub=false

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=roborage/eval_zero_pi05_8 --dataset.num_episodes=2 --dataset.single_task="Plug charger into the socket" --policy.path=lerobot/pi05_base --dataset.episode_time_s=30  --dataset.reset_time_s=10 --dataset.push_to_hub=false

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=roborage/eval_zero_pi05_4 --dataset.num_episodes=2 --dataset.single_task="Plug charger into the socket" --policy.path=lerobot/pi05_base --dataset.episode_time_s=30  --dataset.reset_time_s=10 --dataset.push_to_hub=false 

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ right_wrist_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, base_0_rgb: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=roborage/eval_zero_pi05_10 --dataset.num_episodes=2 --dataset.single_task="Plug charger into the socket" --policy.path=lerobot/pi05_base --dataset.episode_time_s=30  --dataset.reset_time_s=10 --dataset.push_to_hub=false

lerobot-edit-dataset     --repo_id roborage/record_mergedv2     --operation.type merge     --operation.repo_ids "['roborage/record_v2', 'roborage/record_v3_1', 'roborage/record_v4', 'roborage/record_v5', 'roborage/record_v6', 'roborage/record_v7', 'roborage/record_v8', 'roborage/recor
d_v9', 'roborage/record_v10', 'roborage/record_v11']" --push_to_hub true

lerobot-edit-dataset     --repo_id roborage/record_switch_plug_merged     --operation.type merge     --operation.repo_ids "['roborage/record_switch_v1', 'roborage/record_switch_v2']" --push_to_hub true

lerobot-edit-dataset     --repo_id roborage/record_master_merged     --operation.type merge     --operation.repo_ids "['roborage/record_mergedv2', 'roborage/record_switch_merged', roborage/record_switch_plug_merged]" --push_to_hub true

## ACtual commands

# pi05

just pick up

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.repo_id=roborage/eval_whole_pi05_2 --dataset.num_episodes=2 --dataset.single_task="Plug charger into the socket" --policy.path=roborage/pi05_plugging_charger_turnon_rabc --dataset.episode_time_s=30 --dataset.reset_time_s=10 --dataset.push_to_hub=false

whole thing

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' --display_data=true --dataset.num_episodes=2 --dataset.single_task="Turn on the switch and turn on the switch" --policy.path=andlyu/pi0fast_record_v2 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --dataset.push_to_hub=false --dataset.repo_id=roborage/eval_whole_pifast05_5

## for groot

lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' --display_data=true  --dataset.num_episodes=1 --dataset.single_task="Plug charger into the socket and turn on the switch" --policy.path=igkp/groot_100k --dataset.episode_time_s=30  --dataset.reset_time_s=10 --dataset.push_to_hub=false --dataset.repo_id=roborage/eval_groot-100k_1