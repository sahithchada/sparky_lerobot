for running groot vla- (this runs a vla which plugs the charger in and turn on the switch)
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follower_2 --robot.cameras='{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' --display_data=true  --dataset.num_episodes=1 --dataset.single_task="plug charger into the socket" --policy.path=igkp/groot_100k --dataset.episode_time_s=300  --dataset.reset_time_s=300 --dataset.push_to_hub=false --dataset.repo_id=roborage/eval_groot-100k_11

Hard coded code to get the robot to turn on the switch
lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=kanishk_lerobot_follo
wer_2 --dataset.repo_id=roborage/record_switch_v2 --dataset.episode=4