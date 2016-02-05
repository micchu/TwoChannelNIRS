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

# randomly shuffle the main trials
# main_trials.shuffle();
rest.present();
main_trials[1].present();
rest.present();
main_trials[2].present();
rest.present();
main_trials[3].present();
rest.present();
main_trials[4].present();
rest.present();
main_trials[5].present();
rest.present();
main_trials[6].present();
rest.present();
main_trials[7].present();
rest.present();
main_trials[8].present();
rest.present();
main_trials[9].present();
rest.present();
main_trials[10].present();
rest.present();
main_trials[11].present();
rest.present();
main_trials[12].present();
rest.present();
main_trials[13].present();
rest.present();
main_trials[14].present();
rest.present();
main_trials[15].present();
rest.present();
main_trials[16].present();
rest.present();
main_trials[17].present();
rest.present();
main_trials[18].present();
rest.present();
main_trials[19].present();
rest.present();
main_trials[20].present();
rest.present();
main_trials[21].present();
rest.present();
main_trials[22].present();
rest.present();
main_trials[23].present();
rest.present();
main_trials[24].present();
rest.present();
main_trials[25].present();
rest.present();