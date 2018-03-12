// Expand Decision Tree 10
// 进行多次测试实验，去最小的十折交叉验证平均误差率

#include"class12.h"

int main()
{
	cout<<"程序开始运行……"<<endl;

/*	int m_TrainData=4177;
	int m_dim=7;
	int m_Test=4177;
	string train_file("new_abalone.data");
	string test_file("new_abalone.data");  
	string predict_file("predict_new_abalone1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=690;
	int m_dim=6;
	int m_Test=690;
	string train_file("new_australian.data");
	string test_file("new_australian.data");  
	string predict_file("predict_new_australian1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=160;
	int m_dim=15;
	int m_Test=160;
	string train_file("new_autos.data");
	string test_file("new_autos.data");  
	string predict_file("predict_new_autos1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=625;
	int m_dim=4;
	int m_Test=625;
	string train_file("new_balance.data");
	string test_file("new_balance.data");  
	string predict_file("predict_new_balance1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=365;
	int m_dim=18;
	int m_Test=365;
	string train_file("new_bands.data");
	string test_file("new_bands.data");  
	string predict_file("predict_new_bands1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=1372;
	int m_dim=4;
	int m_Test=1372;
	string train_file("new_banknote.data");   
	string test_file("new_banknote.data");  
	string predict_file("predict_new_banknote1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=345;
	int m_dim=6;
	int m_Test=345;
	string train_file("new_bupa.data");
	string test_file("new_bupa.data");  
	string predict_file("predict_new_bupa1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=297;
	int m_dim=13;
	int m_Test=297;
	string train_file("new_cleveland.data");
	string test_file("new_cleveland.data");  
	string predict_file("predict_new_cleveland1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=1473;
	int m_dim=9;
	int m_Test=1473;
	string train_file("new_contraceptive.data");
	string test_file("new_contraceptive.data");  
	string predict_file("predict_new_contraceptive.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=690;
	int m_dim=6;
	int m_Test=690;
	string train_file("new_crx.data");
	string test_file("new_crx.data");  
	string predict_file("predict_new_crx1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=336;
	int m_dim=5;
	int m_Test=336;
	string train_file("new_ecoli.data");
	string test_file("new_ecoli.data");  
	string predict_file("predict_new_ecoli1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=212;
	int m_dim=9;
	int m_Test=212;
	string train_file("new_glass.data");
	string test_file("new_glass.data");  
	string predict_file("predict_new_glass1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=306;
	int m_dim=3;
	int m_Test=306;
	string train_file("new_haberman.data");
	string test_file("new_haberman.data");  
	string predict_file("predict_new_haberman1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=270;
	int m_dim=13;
	int m_Test=270;
	string train_file("new_heart.data");
	string test_file("new_heart.data");  
	string predict_file("predict_new_heart1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=150;
	int m_dim=4;
	int m_Test=150;
	string train_file("new_iris.data");
	string test_file("new_iris.data");  
	string predict_file("predict_new_iris.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=360;
	int m_dim=90;
	int m_Test=360;
	string train_file("new_movement.data");
	string test_file("new_movement.data");  
	string predict_file("predict_new_movement1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=5473;
	int m_dim=10;
	int m_Test=5473;
	string train_file("new_pageblocks.data");
	string test_file("new_pageblocks.data");  
	string predict_file("predict_new_pageblocks1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=10992;
	int m_dim=16;
	int m_Test=10992;
	string train_file("new_penbased.data");
	string test_file("new_penbased.data");  
	string predict_file("predict_new_penbased1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=768;
	int m_dim=8;
	int m_Test=768;
	string train_file("new_pima.data");
	string test_file("new_pima.data");  
	string predict_file("predict_new_pima1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=2310;
	int m_dim=18;
	int m_Test=2310;
	string train_file("new_segmentation.data");
	string test_file("new_segmentation.data");  
	string predict_file("predict_new_segmentation1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=2584;
	int m_dim=10;
	int m_Test=2584;
	string train_file("new_seismic.data");
	string test_file("new_seismic.data");  
	string predict_file("predict_new_seismic1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=208;
	int m_dim=60;
	int m_Test=208;
	string train_file("new_sonar.data");
	string test_file("new_sonar.data");  
	string predict_file("predict_new_sonar1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=4601;
	int m_dim=57;
	int m_Test=4601;
	string train_file("new_spambase.data");
	string test_file("new_spambase.data");  
	string predict_file("predict_new_spambase1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=267;
	int m_dim=44;
	int m_Test=267;
	string train_file("new_specfheart.data");
	string test_file("new_specfheart.data");  
	string predict_file("predict_new_specfheart1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=748;
	int m_dim=4;
	int m_Test=748;
	string train_file("new_transfusion.data");
	string test_file("new_transfusion.data");  
	string predict_file("predict_new_transfusion1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=990;
	int m_dim=10;
	int m_Test=990;
	string train_file("new_vowel.data");
	string test_file("new_vowel.data");  
	string predict_file("predict_new_vowel1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */
	
/*	int m_TrainData=178;
	int m_dim=13;
	int m_Test=178;
	string train_file("new_wine.data");
	string test_file("new_wine.data");  
	string predict_file("predict_new_wine1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=683;
	int m_dim=9;
	int m_Test=683;
	string train_file("new_wisconsin.data");
	string test_file("new_wisconsin.data");  
	string predict_file("predict_new_wisconsin1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=1484;
	int m_dim=6;
	int m_Test=1484;
	string train_file("new_yeast.data");
	string test_file("new_yeast.data");  
	string predict_file("predict_new_yeast1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

	//以下为另一篇文献数据集

/*	int m_TrainData=683;
	int m_dim=8;
	int m_Test=683;
	string train_file("cancer.data");
	string test_file("cancer.data");  
	string predict_file("predict_cancer.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=221;
	int m_dim=36;
	int m_Test=221;
	string train_file("CT.data");
	string test_file("CT.data");  
	string predict_file("predict_CT.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=1000;
	int m_dim=3;
	int m_Test=1000;
	string train_file("german.data");
	string test_file("german.data");  
	string predict_file("predict_german.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=351;
	int m_dim=34;
	int m_Test=351;
	string train_file("ionosphere.data");
	string test_file("ionosphere.data");  
	string predict_file("predict_ionosphere1.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=360;
	int m_dim=90;
	int m_Test=360;
	string train_file("libras.data");
	string test_file("libras.data");  
	string predict_file("predict_libras.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=768;
	int m_dim=8;
	int m_Test=768;
	string train_file("pima.data");
	string test_file("pima.data");  
	string predict_file("predict_pima.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=182;
	int m_dim=12;
	int m_Test=182;
	string train_file("plrx.data");
	string test_file("plrx.data");  
	string predict_file("predict_plrx.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=267;
	int m_dim=44;
	int m_Test=267;
	string train_file("SPECTF.data");
	string test_file("SPECTF.data");  
	string predict_file("predict_SPECTF.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

/*	int m_TrainData=569;
	int m_dim=30;
	int m_Test=569;
	string train_file("wdbc.data");
	string test_file("wdbc.data");  
	string predict_file("predict_wdbc.data");  
	Data<double, string> Da(m_TrainData, m_dim);  */

	int m_TrainData=198;
	int m_dim=33;
	int m_Test=198;
	string train_file("wpbc.data");
	string test_file("wpbc.data");  
	string predict_file("predict_wpbc.data");  
	Data<double, string> Da(m_TrainData, m_dim);  

	clock_t start_time=clock();    //开始计算运行时间
	
	vector<int> ivec;    //存储数据点编号
	Da.input(train_file,ivec);  //读入数据, 归一化  (x-min)/(max-min)
//	Da.input1(train_file,ivec);  //读入数据，正态归一化 (x-mean)/var
//	Da.print();

	//建立输出文件流
	ofstream fout;
	open_outfile(fout,predict_file);

//	int n_times=10;   //交叉验证次数
	int total_num=ivec.size();  //总样本数
	int folds=10;        // 10-folds cross-validation
	int max_n=100;        //最大实验次数
	map<double,performance> crector_map;  //正确率容器,从小到大排序
	for(int k=0; k<max_n;k++)
	{
		vector<int> vec=ivec;
		// ivec 里的数据顺序生成随机序列
		//pointer object to it
		ptrdiff_t (*p_myrandom)(ptrdiff_t)=myrandom;   //函数指针
		// 5 次调用随机序列生成函数，以期将序列打乱
		random_shuffle(vec.begin(),vec.end(),p_myrandom);
		random_shuffle(vec.begin(),vec.end(),p_myrandom);
		random_shuffle(vec.begin(),vec.end(),p_myrandom);
		random_shuffle(vec.begin(),vec.end(),p_myrandom);
		random_shuffle(vec.begin(),vec.end(),p_myrandom);
		random_shuffle(vec.begin(),vec.end(),p_myrandom);

		performance per=fold_cross_validation(folds,vec,Da);
		crector_map.insert(make_pair(per.mean,per));
	}

	clock_t end_time=clock();      //结束计算运行时间

	//采用反向迭代器取出最大正确率的 performance 对象
	map<double,performance>::reverse_iterator r_it=crector_map.rbegin();
	performance per_best=(r_it->second);

	open_outfile2(fout,predict_file);
	fout<<"数据集："<<train_file<<" ; 十折交叉验证最高精度："<<per_best.mean<<" + "<<per_best.standar_var<<endl;
	fout<<"决策树平均深度："<<per_best.depth<<" , 决策树平均总结点数："<<per_best.nodes<<endl;
	fout<<"程序运行总时间："<<running_time(start_time,end_time)<<"s."<<endl;

	cout<<"程序运行结束！"<<endl;
	system("pause");
}