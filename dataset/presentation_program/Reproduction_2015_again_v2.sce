scenario = "Repriduction_2015_again";
no_logfile = false;   
default_font_size = 120;   
default_trial_duration = 6000;#タスク時間の設定(デフォルト：6000=6秒)

begin;
picture {} default;
trial {
   trial_duration = 4000;#レスト時間の設定(デフォルト：4000=4秒)
   picture {
      text { caption = "+"; };
      x = 0; y = 0;
   };
} rest;         

# PCLの配列を作成する
array {
   LOOP $i 25;
   $k = '$i + 1';
   trial {
      picture{
       bitmap{
       filename="$k.jpg";};
       x = 0; y = 0;
    };
      code = "Pic $k";
   };
   ENDLOOP;
} main_trials;

begin_pcl;
int first_bunch = 25;

rest.present();

# random order version 1
array <int>random_order[25] = {10,9,23,15,21,1,12,8,2,14,4,20,25,18,16,3,22,24,17,13,19,11,6,5,7};
# random order version 2
#array <int>random_order[25] = {19,5,12,11,3,7,16,15,21,13,25,6,20,10,9,4,1,14,17,22,2,8,18,23,24};

loop int counter = 1 until counter > 25 begin
	main_trials[random_order[counter]].present();
	rest.present();
	counter = counter + 1;
end;
