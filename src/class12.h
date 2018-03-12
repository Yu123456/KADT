// Expand Decision Tree 12
// 扩展于 Expand Decision Tree 10 
// 扩展于 Expand Decision Tree 8  直接从 data 中读取数据
// 编写于 2015年6月5日
// 修改于 2015年5月29日，增加 10-folds 交叉验证
// k-fold 交叉验证的准确率定义为：the average correct value of the 10 results
//主要思想为隶属度函数及扩展基尼指数
// 修改于 2015年5月22日下午，
// 1、当 sigma^2=0 时，隶属度应该控制在 0.05 以上，这样就可以确保超过 delta/2 的点隶属度小于 0.05
//修改于2015年5月27日，增加决策树深度及结点计算 pair<int,int> count_depth_nodes(); 

// 修改于2015年6月5日，增加原始数据归一化，(x-x_min)/(x_max-x_min)


#ifndef CLASS12_H
#define CLASS12_H

#include<iostream>
#include<fstream>      //文件流
#include<sstream>
#include<ctime>     // 运行时间
#include<set>
#include<vector>
#include<map>
#include<cmath>
#include<queue>
#include<cstdio>
#include<cstdlib>
#include<functional>
#include<algorithm>
#include<iterator>

using namespace std;

//类的前向声明
template<class T, class T2> class Data;
template<class T, class T2> class Node;
template<class T, class T2> class DecisionTree;
//类标签、均值、方差类
template<class T,class T2> class c_MuSigma2;
//测试数据集类
template<class T,class T2> class Tdata;
//性能评估类，主要是保存十折交叉验证平均值、标准差、平均树深度、平均结点数
class performance;

//非类成员函数

//运行时函数
double running_time(const clock_t &s, const clock_t &e)
{
	return static_cast<double>(e-s)/CLOCKS_PER_SEC;
}

//打开输入文件流函数
ifstream& open_file(ifstream &in, const string &file)
{
	in.close();
	in.clear();
	in.open(file.c_str());
	return in;
}

//打开输出文件流函数
ofstream& open_outfile(ofstream &out, const string &file)
{
	out.close();
	out.clear();
	out.open(file.c_str(), ofstream::trunc);  //清空原有文件
	return out;
}

//打开输出文件流函数
ofstream& open_outfile2(ofstream &out, const string &file)
{
	out.close();
	out.clear();
	out.open(file.c_str(), ofstream::app);  //追加到文件末尾
	return out;
}

//字符串转换函数
void string_replace(string &st, const string &pre_str,const string &post_str)
{
	string::size_type pos=0;
	while((pos=st.find_first_of(pre_str,pos))!=string::npos)
	{
		st.replace(pos,1,post_str);
		pos++;
	}
}

//random generator function
ptrdiff_t myrandom(ptrdiff_t i)
{
	srand(unsigned(time(NULL)));
	return rand()%i;
}

//非类成员函数，计算均值、标准差
pair<double,double> Mean_Var(const vector<double> &vec)
{
	double mean=0.0,var=0.0;
	int n=vec.size();
	for(vector<double>::const_iterator it=vec.begin();it!=vec.end();it++)
	{
		double value=(*it);
		mean+=value;
		var+=value*value;
	}
	mean=mean/double(n);
	var=var/double(n)-mean*mean;
	var=sqrt(var);
	return make_pair(mean,var);
}


//性能评估类，主要是保存十折交叉验证平均值、标准差、平均树深度、平均结点数
class performance
{
public:
	double mean;          //正确率平均值
	double standar_var;   //正确率标准差
	double depth;          //树的平均深度
	double s_depth;        //树的标准差
	double nodes;          //树的平均结点数
	double s_nodes;        //结点数标准差

	performance(){}      //默认构造函数
	performance(double p_m, double p_v, double p_d, double p_ds,double p_n,double p_ns):mean(p_m),
		standar_var(p_v),depth(p_d),s_depth(p_ds),nodes(p_n),s_nodes(p_ns){}
};

// k-fold cross-validation 函数，返回正确率平均值及标准差 ,树的深度，平均结点数
// folds 交叉验证折数，vec 数据编号容器
template<class T,class T2> performance fold_cross_validation(int folds, const vector<int> &vec,  Data<T,T2> &Da)
{
	int m_TrainData=vec.size();
	int m=m_TrainData/folds;   //每一折数据点数，最后一折可能会多于这个
	vector<double> v_rate;     //记录每一折正确率
	vector<double> v_depth;
	vector<double> v_nodes;
	//前folds-1次验证集/测试集数据一样多
	for(int i=0; i<folds-1; i++)
	{
		//验证集
		vector<int> vec_test(vec.begin()+i*m,vec.begin()+(i+1)*m);
		//输出测试集测试是否有问题
		//for(vector<int>::iterator it=vec_test.begin();it!=vec_test.end();it++)
			//cout<<*it<<endl;
		//cout<<"总数："<<vec_test.size()<<endl;
		//训练集
		vector<int> vec_train(vec.begin(),vec.begin()+i*m);
		vector<int> vec_train2(vec.begin()+(i+1)*m,vec.end());
		vec_train.insert(vec_train.end(),vec_train2.begin(),vec_train2.end());
		//建立决策树
		DecisionTree<T,T2> DTree(vec_train,Da);
		pair<int,int> pii=DTree.count_depth_nodes();
		v_depth.push_back(pii.first);
		v_nodes.push_back(pii.second);
		//建立测试集对象
		Tdata<T,T2> Ta(vec_test.size(),Da.dimension);
		Ta.input(vec_test,Da);
		//进行决策
		DTree.MakeDecision(Ta);
		v_rate.push_back(Ta.crector_rate());   //计算正确率
		//清空相关容器
		vec_test.clear();
		vec_train.clear();
		vec_train2.clear();
	}
	// 第 folds 次交叉验证
	//验证集
	vector<int> vec_test(vec.begin()+(folds-1)*m,vec.end());
	//输出测试集测试是否有问题
	//for(vector<int>::iterator it=vec_test.begin();it!=vec_test.end();it++)
		//cout<<*it<<endl;
	//cout<<"总数："<<vec_test.size()<<endl;
	//训练集
	vector<int> vec_train(vec.begin(),vec.begin()+(folds-1)*m);
	//建立决策树
	DecisionTree<T,T2> DTree(vec_train,Da);
	pair<int,int> pii=DTree.count_depth_nodes();
	v_depth.push_back(pii.first);
	v_nodes.push_back(pii.second);
	//建立测试集对象
	Tdata<T,T2> Ta(vec_test.size(),Da.dimension);
	Ta.input(vec_test,Da);
	//进行决策
	DTree.MakeDecision(Ta);
	v_rate.push_back(Ta.crector_rate());    //计算正确率
	//清空容器
	vec_test.clear();
	vec_train.clear();
	pair<double,double> re_p=Mean_Var(v_rate);
	pair<double,double> re_p2=Mean_Var(v_depth);
	pair<double,double> re_p3=Mean_Var(v_nodes);

	return performance(re_p.first,re_p.second,re_p2.first,re_p2.second,re_p3.first,re_p3.second);
}

//非类成员函数声明
//递归构建决策树函数
// remain_point 数据集中剩余样本点编号
// remain_attribute 数据集中剩余样本属性，不含类标号
// da 完整数据集
template<class T,class T2> Node<T,T2>* BuildDecisionTree(Node<T,T2> *, vector<int> ,vector<string> , Data<T,T2> & );

// T 数据属性值类型，T2 数据类标签类型
template<class T, class T2> class Data
{
public:
	int num;      //样本点总数
	int dimension;    //样本维数/属性个数(不含类标号)
	T **data;     //总数据，不含类标号
	T2 *label;     //总数据类编号，与 data 中对应
	string *dim_label;   //属性标签，存储各个属性名称,最后一列为类标签
	T *max_T;       //各个属性的最大值，用于归一化处理    // 在 input1 中保存标准差
	T *min_T;        //各个属性的最小值，用于归一化处理    // 在 input1 中保存均值

	Data(int p_num, int p_dimension):num(p_num),dimension(p_dimension)   //构造函数
	{
		data=new T*[num];     //动态分配内存空间
		for(int i=0; i<num; i++)
			data[i]=new T[dimension];
		dim_label=new string[dimension];
		label=new T2[num];
		max_T=new T[dimension];
		min_T=new T[dimension];
	}
	~Data()   //析构函数
	{
		for(int i=0; i<num; i++)
			delete [] data[i];
		delete [] data;
		delete [] dim_label;
		delete [] label;
		delete [] max_T;
		delete [] min_T;
	}
	void input(string, vector<int>&);    //数据读入函数, 归一化 (x-x_min)/(max-min)
	void input1(string, vector<int>&);  //数据读入函数，正态归一化  (x-mean)/var
	void print();          //输出测试函数
	bool IsAllTheSameLabel(const vector<int>&);  //判断是否属于同一类
	//根据样本编号，返回该样本的类标签
	T2 return_label(int i)
	{
		return label[i];
	}
	//返回多数类标签
	T2 MostCommonLabel(const vector<int>&);
	//返回最优属性
	string Attribute_selection_method(const vector<int>& ,const vector<string>& );
	//返回形参属性所对应的编号
	int return_dim_label(const string&);
	//数据类成员函数，计算均值、方差函数
	pair<T,T> MS_f(const vector<int> &, const int );
	//数据类成员函数，当sigma2=0.0时，取隶属度大于等于 0.95 时的 sigma2
	T min_sigma2(const T2 &,const vector<int> &, const int &,const int &);
	//数据类成员函数，计算 Expand Gini Index
	double Expand_Gini_Index(const vector<int> &, const vector<c_MuSigma2<T,T2> > &,const int &);
	//计算 Gini Index
	double Gini(const vector<int>&);
};

//数据类，数据读入函数, ivec 容器将存储数据点编号，从 0 开始计算，与数组相对应
template<class T,class T2> void Data<T,T2>::input(string filename, vector<int> &ivec)
{
	ifstream in;
	open_file(in, filename);  //打开文件
	//数据第一行为属性行
	string in_str;
	getline(in, in_str);   //读入第一行
	string_replace(in_str,",","\t");   // 逗号转换成 "\t"
	istringstream str(in_str);   //字符串流
	string out_str;
	for(int i=0; i<dimension; i++)   //类标签
	{
		str>>out_str;
		dim_label[i]=out_str;
	}
	//由于数据存储于数组中，涉及数组指标，采用 for 循环
	T d_T;     // 数据变量
	T2 d_T2;    //类标号变量
	bool b_flag=true;     //标记第一个数据，首先保存到 max_T 中
	for(int j=0; j<num; j++) //开始读入数据
	{
		getline(in, in_str);
		string_replace(in_str,",","\t");   // 逗号转换成 "\t"
		istringstream str(in_str);
		if(b_flag)
		{
			for(int i=0; i<dimension; i++)
			{
				str>>d_T;
				data[j][i]=d_T;
				max_T[i]=d_T;
				min_T[i]=d_T;
			}
			b_flag=false;
		}
		else
		{
			for(int i=0; i<dimension; i++)
			{
				str>>d_T;
				data[j][i]=d_T;
				if(d_T>max_T[i])   //保存最大值
					max_T[i]=d_T;
				if(d_T<min_T[i])   //保存最小值
					min_T[i]=d_T;
			}
		}
		str>>d_T2;
		label[j]=d_T2;
		ivec.push_back(j);
	}
	//进行归一化
	T *va=new T[dimension];
	for(int i=0;i<dimension;i++)
		va[i]=max_T[i]-min_T[i];
	for(int j=0; j<num; j++)
	{
		for(int i=0; i<dimension; i++)
			data[j][i]=(double(data[j][i])-double(min_T[i]))/double(va[i]);
	}
	delete [] va;
}


//数据类，数据读入函数, ivec 容器将存储数据点编号，从 0 开始计算，与数组相对应
//注意，此处将 min_T 保存均值，max_T 保存标准差
template<class T,class T2> void Data<T,T2>::input1(string filename, vector<int> &ivec)
{
	ifstream in;
	open_file(in, filename);  //打开文件
	//数据第一行为属性行
	string in_str;
	getline(in, in_str);   //读入第一行
	string_replace(in_str,",","\t");   // 逗号转换成 "\t"
	istringstream str(in_str);   //字符串流
	string out_str;
	for(int i=0; i<dimension; i++)   //类标签
	{
		str>>out_str;
		dim_label[i]=out_str;
	}
	//由于数据存储于数组中，涉及数组指标，采用 for 循环
	T d_T;     // 数据变量
	T2 d_T2;    //类标号变量
	for(int i=0;i<dimension;i++)
	{
		min_T[i]=0.0;
		max_T[i]=0.0;
	}
	for(int j=0; j<num; j++) //开始读入数据
	{
		getline(in, in_str);
		string_replace(in_str,",","\t");   // 逗号转换成 "\t"
		istringstream str(in_str);
		for(int i=0; i<dimension; i++)
		{
			str>>d_T;
			data[j][i]=d_T;
			min_T[i]+=d_T;
			max_T[i]+=d_T*d_T;
		}
		str>>d_T2;
		label[j]=d_T2;
		ivec.push_back(j);
	}
	//计算均值，标准差
	for(int i=0; i<dimension; i++)
	{
		min_T[i]=min_T[i]/double(num);                              // 均值
		max_T[i]=sqrt(max_T[i]/double(num)-min_T[i]*min_T[i]);    // 标准差
	}
	//正态标准化 
	for(int j=0; j<num; j++)
	{
		for(int i=0; i<dimension; i++)
			data[j][i]=(data[j][i]-min_T[i])/max_T[i];
	}
}


//数据类输出测试函数
template<class T, class T2> void Data<T,T2>::print()
{
	cout<<"输出结果："<<endl;
	int j=0;
	for(j=0; j<dimension; j++)
		cout<<dim_label[j]<<"\t";
	cout<<"class label"<<endl;
	for(int i=0; i<num; i++)
	{
		for(j=0; j<dimension; j++)
		{
			cout<<data[i][j]<<"\t";
		}
		cout<<label[i]<<endl;
	}
	cout<<"最大值为："<<endl;
	for(int i=0; i<dimension; i++)
		cout<<max_T[i]<<"\t";
	cout<<endl;
	cout<<"最小值为："<<endl;
	for(int i=0; i<dimension; i++)
		cout<<min_T[i]<<"\t";
	cout<<endl;
}

//数据类成员函数
//判断样本是否属于同一类，相同返回 true, 否则返回 false
template<class T,class T2> bool Data<T,T2>::IsAllTheSameLabel(const vector<int> &vec)
{
	T2 v_label=label[vec[0]];
	for(vector<int>::const_iterator it=vec.begin(); it!=vec.end(); it++)
	{
		if(v_label != label[*it])
			return false;
	}
	return true;
}

//数据点类成员函数，返回多数类标签
template<class T, class T2> T2 Data<T,T2>::MostCommonLabel(const vector<int> &vec)
{
	map<T2,int> map_count;
	for(vector<int>::const_iterator it=vec.begin(); it!=vec.end(); it++)
		++map_count[label[*it]];
	map<T2,int>::iterator iter=map_count.begin();
	T2 str=iter->first;
	int integer=iter->second;
	for( ; iter!=map_count.end(); iter++)
	{
		if(iter->second > integer)
		{
			str=iter->first;
			integer=iter->second;
		}
	}
	return str;
}

//数据点类成员函数，返回最优属性
// ivec 数据点编号，对应总样本中的编号，svec 属性集
template<class T,class T2>	string Data<T,T2>::Attribute_selection_method(const vector<int> &ivec ,const vector<string> &svec )
{
	if(svec.size()==1)
		return svec[0];
	else
	{
		map<double,string> sd_map;   // map< Expand Gini Index,attribute> ，这样可以利用 map 的按照键值升序保存而直接取第一个属性
		multimap<T2,int> ti_map;     // 将数据按类分开，保存数据编号
		set<T2> t_set;         //保存类标签
		typedef multimap<T2,int>::size_type sz_type;   
		for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
		{
			T2 label(label[*it]);
			ti_map.insert(make_pair(label, *it));
			t_set.insert(label);
		}
		for(vector<string>::const_iterator iter=svec.begin(); iter!=svec.end(); iter++)  //计算每一个属性
		{
			string s_attribute=(*iter);    //属性
			int dim_i=return_dim_label(s_attribute);   //返回该属性在数据集 data 中所对应的标号
			vector<int> new_ivec;       //存储每一个类下的数据编号
			vector<c_MuSigma2<T,T2> > cvec;   // 存储类标签、均值、方差
			for(set<T2>::iterator it=t_set.begin(); it!=t_set.end();it++)   //计算每一个类
			{
				T2 t_label=(*it);  //类标签
				//将数据编号按照类标签 t_label 存储在 new_ivec 中，用于计算 mu, sigma2
				sz_type entries=ti_map.count(t_label);    // t_label 键的个数
				multimap<T2,int>::iterator f_it=ti_map.find(t_label);   // t_label 的起始迭代器
				for(sz_type i=0; i!=entries; i++, f_it++)  // t_label 类的数据保存到 new_ivec 中
					new_ivec.push_back(f_it->second);
				pair<T,T> MuSigma2=MS_f(new_ivec,dim_i);
				if(abs(MuSigma2.second-0.0)<1e-4)
					MuSigma2.second=min_sigma2(t_label,ivec,dim_i,new_ivec[0]);// 调用函数计算最短距离，计算 sigma2
				cvec.push_back(c_MuSigma2<T,T2>(t_label,MuSigma2.first,MuSigma2.second));   //存储
				new_ivec.clear();   //清空 new_ivec ，存储下一个类的数据编号
			}
			//计算扩展基尼指数 Expand Gini Index
			double EGI=Expand_Gini_Index(ivec,cvec,dim_i);
			sd_map.insert(make_pair(EGI,s_attribute));
			cvec.clear();   //清空隶属度参数容器
		}
		return sd_map.begin()->second;
	}
}


//返回形参属性所对应的编号
template<class T,class T2> int Data<T,T2>::return_dim_label(const string &str)
{
	for(int i=0; i<dimension; i++)
	{
		if(str==dim_label[i])
			return i;
	}
	return -1;
}

//数据类成员函数，计算均值、方差函数
// ivec 数据编号， dim 属性编号
template<class T,class T2> pair<T,T> Data<T,T2>::MS_f(const vector<int> &ivec, const int dim)
{
	T mean=0.0, var=0.0;
	int n=ivec.size();
	for(int i=0; i<n; i++)
	{
		int k=ivec[i];
		mean+=data[k][dim];
		var+=data[k][dim]*data[k][dim];
	}
	mean=mean/n;
	var=var/n-mean*mean;   //sigma^2
	return make_pair(mean,var);
}

//数据类成员函数，当sigma2=0.0时，取隶属度大于等于 0.95 时的 sigma2
// t_label 当前类类标签， ivec 数据集编号，dim 属性编号，kk 当前类的一个数据样本编号
template<class T,class T2> T Data<T,T2>::min_sigma2(const T2 &t_label,const vector<int> &ivec, const int &dim,const int &kk)
{
	T dis,dis2;
	T d_target=data[kk][dim];
	bool flag=true;
	for(unsigned int i=0; i<ivec.size(); i++)
	{
		int k=ivec[i];
		if(label[k]!=t_label)
		{
			if(flag)
			{
				dis=abs(data[k][dim]-d_target);
				flag=false;
			}
			else
			{
				dis2=abs(data[k][dim]-d_target);
				if(dis>dis2)
					dis=dis2;
			}
		}
	}
	if(dis<=0.001)
		return 0.0001;
	else
	{
		//return 2.4369657*dis*dis;   // 此时 delta/2 隶属度控制在 0.95
	   // return 1.186402698*dis*dis;  //此时 delta/2 隶属度控制在 0.9
	   return 0.041726*dis*dis;      //此时 delta/2 隶属度控制在 0.05
		//return 0.166904*dis*dis;       // 此时 delta 隶属度控制在 0.05
	}
}

//数据类成员函数，计算 Gini Index
template<class T,class T2> double Data<T,T2>::Gini(const vector<int> &ivec)
{
	if(ivec.empty())
		return 0.0;
	typedef map<T2,int> map_s_i;   // <类标签，int>
	map_s_i smap;
	for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
		++smap[label[*it]];
	double value=0.0;
	for(map_s_i::iterator it=smap.begin(); it!=smap.end(); it++)
		value+=double(it->second)*double(it->second);
	return 1.0-value/(double(ivec.size())*double(ivec.size()));
}

//数据类成员函数，计算 Expand Gini Index
// ivec 数据标号，cvec 类的隶属度参数, dim 属性标号
template<class T,class T2> double Data<T,T2>::Expand_Gini_Index(const vector<int> &ivec, const vector<c_MuSigma2<T,T2> > &cvec,const int &dim)
{
	int m=cvec.size();    //类的个数
	vector<int> *p=new vector<int>[m];   //容器数组，这里的顺序与 vector<c_MuSigma2<T,T2> > 对应上，保存隶属度最大时的数据编号
	//按照隶属度最大原则，将数据标号保存在 p 数组中
	for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
	{
		int k=(*it);
		T var=data[k][dim];
		double best_menbership_degree=cvec[0].Menbership(var);//返回隶属度
		int best_flag=0;        //隶属度函数标记，标记该隶属度函数在 cvec 中的标号
		for(unsigned int i=1; i<cvec.size(); i++)
		{
			double bmd=cvec[i].Menbership(var);
			if(bmd>best_menbership_degree)
			{
				best_menbership_degree=bmd;
				best_flag=i;
			}
		}
		//保存数据编号
		p[best_flag].push_back(k);
	}
	//计算 Expand Gini Index
	double egi=0.0;
	for(int i=0; i<m; i++)
	{
		egi+=double(p[i].size())*Gini(p[i]);
	}
	delete [] p;   //释放动态空间
	return egi/double(ivec.size());
}


//类标签、均值、方差类
template<class T,class T2> class c_MuSigma2
{
public:
	T2 class_label;
	T mu;
	T sigma2;

	c_MuSigma2(){}    //默认构造函数
	c_MuSigma2(T2 c, T m,T s):class_label(c),mu(m),sigma2(s){}    //构造函数
	const double Menbership(const T &) const;    //隶属度计算函数, const 对象调用
	double Menbership(const T &);    //重载隶属度计算函数
};

//均值、方差类成员函数，计算隶属度
template<class T,class T2> const double c_MuSigma2<T,T2>::Menbership(const T &var) const
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}

//均值、方差类成员函数，计算隶属度
template<class T,class T2> double c_MuSigma2<T,T2>::Menbership(const T &var)
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}


//结点类
// T 数据值类型
// T2 类标签类型
template<class T, class T2> class Node
{
public:
	string attribute;  //属性值
	T mu;    //高斯隶属度函数的均值
	T sigma2;  //高斯隶属度函数的方差  sigma^2
	T2 class_label;   //类标签
	vector<Node<T,T2>* > childs;   //所有孩子

	Node(){}   //默认构造函数
	bool IsLeaf()   //判断是否为树叶
	{
		if(childs.empty())
			return true;
		else
			return false;
	}
	//结点类成员函数，计算隶属度
	double Menbership(T );
};

//结点类成员函数，计算隶属度
template<class T,class T2> double Node<T,T2>::Menbership(T var)
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}


//决策树类
// T 数据值类型
// T2 类标签类型
template<class T, class T2> class DecisionTree
{
public:
	Node<T,T2> *root;

	//成员函数
	//DecisionTree():root(NULL){}    //默认构造函数
	DecisionTree(vector<int> ,Data<T,T2> &);       // 以数据集建立决策树
	void FreeTree(Node<T,T2> *p)
	{
		if(p != NULL )
		{
			for(vector<Node<T,T2>* >::iterator it=p->childs.begin(); it!=p->childs.end(); it++)
				FreeTree(*it);
			delete p;
		}
	}
	// 析构函数
	~DecisionTree(){ FreeTree(root); }
	//输出决策树
	void PrintDT();
	//输出到文件里
	void PrintDT(ofstream &,const string &);
	//决策函数
	void MakeDecision(Tdata<T,T2> &);
	T2 recursion_MD(Node<T,T2> *, Tdata<T,T2> &, int);  //供决策函数递归调用决策
	pair<int,int> count_depth_nodes();        //计算 Decision Tree 深度及结点个数  pair<depth,nodes>
};

//输出到文件里
template<class T,class T2> void DecisionTree<T,T2>::PrintDT(ofstream &fout, const string &str)
{
	queue<Node<T,T2>* > que;
	Node<T,T2> *pointer=root;
	if(pointer)
		que.push(pointer);
	int current_n=1, next_n=0, level_n=0;
	fout<<"Expand Decision Tree"<<endl;
	fout<<" 第 "<<level_n<<" 层树："<<endl;
	pointer=que.front();
	que.pop();
	fout<<"中间结点：attribute: "<<pointer->attribute<<" , 子树："<<pointer->childs.size()<<endl;
	for(vector<Node<T,T2>* >::iterator it=pointer->childs.begin(); it!=pointer->childs.end(); it++)
	{
		que.push(*it);
		next_n++;
	}
	current_n=next_n;
	next_n=0;
	fout<<endl;
	level_n++;
	fout<<" 第 "<<level_n<<" 层树："<<endl;
	while(!que.empty())
	{
		pointer=que.front();
		que.pop();
		if(pointer->IsLeaf())
			fout<<"叶结点： class label: "<<pointer->class_label<<" , mu= "<<pointer->mu<<" , sigma2= "<<pointer->sigma2<<"\t";
		else
			fout<<"中间结点：attribute: "<<pointer->attribute<<" , mu= "<<pointer->mu<<" , sigma2= "<<pointer->sigma2<<" ,子树："<<pointer->childs.size()<<"\t";
		for(vector<Node<T,T2>* >::iterator it=pointer->childs.begin(); it!=pointer->childs.end(); it++)
		{
			que.push(*it);
			next_n++;
		}
		if(!--current_n)
		{
			current_n=next_n;
			next_n=0;
			level_n++;
			fout<<endl;
			fout<<endl;
			if(current_n!=0)
				fout<<" 第 "<<level_n<<" 层树："<<endl;
		}
	}
	fout<<endl;
}

//决策函数
template<class T,class T2> void DecisionTree<T,T2>::MakeDecision(Tdata<T,T2> &td)
{
	Node<T,T2> *p=root;
	//获取测试集中与决策树中属性相一致的对应分类
	int t_num=td.num;
	for(int i=0; i<t_num; i++)
		td.label[i][1]=recursion_MD(root,td,i);
}

//供决策函数递归调用决策
template<class T,class T2> T2 DecisionTree<T,T2>::recursion_MD(Node<T,T2> *p, Tdata<T,T2> &td, int i)
{
	if(p->IsLeaf())
		return p->class_label;
	else
	{
		string attribute=p->attribute;    //获取分裂属性
		int i_dim=td.re_dim_label(p->attribute);  //返回所要测试的属性的维数值
		T var=td.data[i][i_dim];    //属性值

		//选取最大隶属度属性
		double menbership_degree=p->childs[0]->Menbership(var);
		unsigned int flag=0;    //标记使用第几个隶属度函数，即选择哪一个属性
		for(unsigned int j=1; j<p->childs.size(); j++)
		{
			double men=p->childs[j]->Menbership(var);
			if(menbership_degree<men)
			{
				menbership_degree=men;
				flag=j;
			}
		}
		return recursion_MD(p->childs[flag],td,i);
	}
}



//构造函数
// remain_point 数据集中剩余样本点编号
// remain_attribute 数据集中剩余样本属性，不含类标号
// da 完整数据集，可以利用 remain_point 里的编号获取完整样本点
template<class T, class T2> DecisionTree<T,T2>::DecisionTree(vector<int> remain_point,Data<T,T2> &da )
{
	root=new Node<T,T2>();
	// 刚开始需要输入所有属性
	vector<string> remain_attribute(da.dim_label, da.dim_label+da.dimension);   
	root=BuildDecisionTree(root, remain_point, remain_attribute, da);
}

//递归构建决策树函数
// remain_point 数据集中剩余样本点编号
// remain_attribute 数据集中剩余样本属性，不含类标号
// da 完整数据集
template<class T,class T2> Node<T,T2>* BuildDecisionTree(Node<T,T2> *p, vector<int> remain_point,
											vector<string> remain_attribute, Data<T,T2> &da )
{
	if(p==NULL)      //如果 Node 指针结点为空，则创建一个
		p=new Node<T,T2>();
	//如果 remain_point 中所有点同一类，此时 Node 为叶结点，标记类标号
	if(da.IsAllTheSameLabel(remain_point))
	{
		//返回类标签
		p->class_label=da.return_label(remain_point[0]);
		return p;
	}
	//如果 remain_attribute 里的属性已经考虑完毕，仍有未分类的样本点，用多数类标签标记该类
	if(remain_attribute.empty())
	{
		//返回多数类标签
		p->class_label=da.MostCommonLabel(remain_point);
		return p;
	}
	//找出最好的属性
	string best_attribute=da.Attribute_selection_method(remain_point,remain_attribute);
	// 用 best_attribute 标记结点
	p->attribute=best_attribute;
	//更新属性集，删除 best_attribute 属性值
	vector<string> new_remain_attribute;
	for(vector<string>::iterator it=remain_attribute.begin(); it!=remain_attribute.end(); it++)
	{
		if(*it !=best_attribute)
			new_remain_attribute.push_back(*it);
	}
	//属性 best_attribute 下的分裂准则，即每个类的隶属度函数
	multimap<T2,int> ti_map;     // 将数据按类分开，保存数据编号
	set<T2> t_set;         //保存类标签
	typedef multimap<T2,int>::size_type sz_type;   
	for(vector<int>::iterator it=remain_point.begin(); it!=remain_point.end(); it++)
	{
		T2 label(da.label[*it]);
		ti_map.insert(make_pair(label, *it));
		t_set.insert(label);
	}
	//计算最优属性下的均值、方差
	int best_dim=da.return_dim_label(best_attribute);   //最优属性对应编号
	vector<int> new_ivec;   //保存每个 splitting_criterion (类) 下的数据点编号
	vector<c_MuSigma2<T,T2> > cvec;   //保存类标签、均值、方差，即 splitting_criterion
	for(set<T2>::iterator it=t_set.begin();it !=t_set.end(); it++)
	{
		T2 t_label=(*it);  //类标签
		//将数据编号按照类标签 t_label 存储在 new_ivec 中，用于计算 mu, sigma2
		sz_type entries=ti_map.count(t_label);   // t_label 键的个数
		multimap<T2,int>::iterator f_it=ti_map.find(t_label);
		// t_label 类的数据保存到 new_ivec 中
		for(sz_type i=0; i!=entries; i++, f_it++)
			new_ivec.push_back(f_it->second);
		//计算均值、方差
		pair<T,T> MuSigma2=da.MS_f(new_ivec,best_dim);
		if(abs(MuSigma2.second-0.0)<1e-4)  //sigma2=0 时，重新赋值 sigma2
			MuSigma2.second=da.min_sigma2(t_label,remain_point,best_dim,new_ivec[0]);
		cvec.push_back(c_MuSigma2<T,T2>(t_label,MuSigma2.first,MuSigma2.second));  //保存分裂隶属度函数
		new_ivec.clear();  // 清空 new_ivec, 存储下一个类的数据编号
	}
	//按照隶属度最大化原则，划分数据
	int cm=cvec.size();   //类的个数，
	vector<int> *data_p=new vector<int>[cm]; //容器数组，这里的顺序与 vector<c_MuSigma2<T,T2> > 对应上，保存隶属度最大时的数据编号
	//按照隶属度最大原则，将数据标号保存在 data_p 数组中
	for(vector<int>::iterator it=remain_point.begin(); it!=remain_point.end(); it++)
	{
		int k=(*it);
		T var=da.data[k][best_dim];
		double best_menbership_degree=cvec[0].Menbership(var);  //返回隶属度
		int best_flag=0;   //隶属度函数标记，标记该函数在 cvec 中的标号
		for(unsigned int i=1; i<cvec.size(); i++)
		{
			double bmd=cvec[i].Menbership(var);
			if(bmd>best_menbership_degree)
			{
				best_menbership_degree=bmd;
				best_flag=i;
			}
		}
		//保存数据编号
		data_p[best_flag].push_back(k);
	}
	//建立分支
	for(int i=0; i<cm; i++)
	{
		Node<T,T2> *new_node=new Node<T,T2>();
		new_node->mu=cvec[i].mu;
		new_node->sigma2=cvec[i].sigma2;
		if(data_p[i].empty())   //当前分支没有样例，即该隶属度函数下没有数据点
		{
			//将该类标记为分裂类
			new_node->class_label=cvec[i].class_label;
			//返回多数类标签
			//new_node->class_label=da.MostCommonLabel(remain_point);
		}
		else
		{
			BuildDecisionTree(new_node,data_p[i],new_remain_attribute,da);
		}
		p->childs.push_back(new_node);
	}
	delete [] data_p;
	cvec.clear();
	return p;
}

//决策树成员函数，计算决策树的深度及结点个数, pair<depth,nodes>
template<class T,class T2> pair<int,int> DecisionTree<T,T2>::count_depth_nodes()
{
	queue<Node<T,T2>* > que;
	Node<T,T2> *pointer=root;
	if(pointer)
		que.push(pointer);
	int current_n=1, next_n=0, level_n=-1, total_n=1;
	while(!que.empty())
	{
		pointer=que.front();
		que.pop();
		for(vector<Node<T,T2>* >::iterator it=pointer->childs.begin(); it!=pointer->childs.end(); it++)
		{
			que.push(*it);
			next_n++;
			total_n++;   //总结点树统计
		}
		if(!--current_n)
		{
			current_n=next_n;
			next_n=0;
			level_n++;
		}
	}
	return make_pair(level_n,total_n);
}


//测试数据集类
template<class T,class T2> class Tdata
{
public:
	int num;  //测试样本总数
	int dimension;  //样本维数，不含标签
	T **data;  //总数据
	T2 **label;  //总数据标签，第二列保存预测类标签
	string *dim_label;  //属性标签，不含类标签

	Tdata(int,int);   //构造函数
	~Tdata()
	{
		delete [] dim_label;
		for(int i=0; i<num; i++)
		{
			delete [] data[i];
			delete [] label[i];
		}
		delete data;
		delete label;
	}
	void input(const string&);  //读入测试集
	void input(const vector<int>&, const Data<T,T2>&);   //从容器中获取数据编号读取数据
	void Print();  //打印测试集
	void Print2();   //打印预测集
	void Print2(ofstream &, const string&);  //打印预测集到文件中
	int re_dim_label(const string &);     //返回对应属性下的维数
	double crector_rate();       //计算预测正确率
	int crector_count();         //统计验证正确数据个数
};

//构造函数
template<class T,class T2> Tdata<T,T2>::Tdata(int p_num, int p_dim):num(p_num),dimension(p_dim)
{
	data=new T*[num];
	label=new T2*[num];
	for(int i=0;i<num; i++)
	{
		data[i]=new T[dimension];
		label[i]=new T2[2];
	}
	dim_label=new string[dimension];
}

//Tdata 成员函数，读入测试集
template<class T,class T2> void Tdata<T,T2>::input(const string &str)
{
	ifstream in;
	open_file(in,str);
	//数据第一行为属性
	string in_str;  
	getline(in,in_str);  //读入第一行
	istringstream istr(in_str);   //字符串流
	string out_str;
	for(int i=0; i<dimension; i++)
	{
		istr>>out_str;
		dim_label[i]=out_str;
	}
	T d_T;   //数据变量
	T2 d_T2;  //类标签变量
	for(int i=0; i<num; i++)
	{
		getline(in,in_str);
		istringstream istr(in_str);
		for(int j=0; j<dimension; j++)
		{
			istr>>d_T;
			data[i][j]=d_T;
		}
		istr>>d_T2;
		label[i][0]=d_T2;
	}
}

//从容器中获取数据编号读取数据
template<class T,class T2> void Tdata<T,T2>::input(const vector<int> &vec, const Data<T,T2> &da)
{
	for(int i=0; i<dimension; i++)   //属性
	{
		dim_label[i]=da.dim_label[i];
	}
	int dim=0;
	for(vector<int>::const_iterator it=vec.begin(); it!=vec.end(); it++)
	{
		int k=(*it);
		for(int j=0; j<dimension; j++)
		{
			data[dim][j]=da.data[k][j];
		}
		label[dim][0]=da.label[k];
		dim++;
	}
}

//打印测试集
template<class T,class T2> void Tdata<T,T2>::Print()
{
	cout<<"输出测试集！"<<endl;
	for(int i=0; i<dimension; i++)
		cout<<dim_label[i]<<"\t";
	cout<<"c_label"<<endl;
	for(int i=0;i<num; i++)
	{
		for(int j=0; j<dimension;j++)
			cout<<data[i][j]<<"\t";
		cout<<label[i][0]<<endl;
	}
}

//打印预测集
template<class T,class T2> void Tdata<T,T2>::Print2()
{
	cout<<"输出预测结果！"<<endl;
	for(int i=0; i<dimension; i++)
		cout<<dim_label[i]<<"\t";
	cout<<"c_label"<<endl;
	for(int i=0;i<num; i++)
	{
		for(int j=0; j<dimension;j++)
			cout<<data[i][j]<<"\t";
		cout<<label[i][0]<<"\t"<<label[i][1]<<endl;
	}
}

//打印预测集到文件中
template<class T,class T2> void Tdata<T,T2>::Print2(ofstream &fout,const string &str)
{
	fout<<endl;
	fout<<"输出预测结果！"<<endl;
	for(int i=0; i<dimension; i++)
		fout<<dim_label[i]<<"\t";
	fout<<"c_label"<<"\t"<<"predict_label"<<endl;
	for(int i=0;i<num; i++)
	{
		for(int j=0; j<dimension;j++)
			fout<<data[i][j]<<"\t";
		fout<<label[i][0]<<"\t"<<label[i][1]<<endl;
	}
}

//返回对应属性下的维数
template<class T,class T2> int Tdata<T,T2>::re_dim_label(const string &str)
{
	for(int i=0; i<dimension; i++)
		if(str==dim_label[i])
			return i;
	return -1;
}

//计算正确率
template<class T,class T2> double Tdata<T,T2>::crector_rate()
{
	int count_true=0, count_false=0;
	for(int i=0; i<num; i++)
	{
		if(label[i][0]==label[i][1])
			count_true++;
		else
			count_false++;
	}
	return double(count_true)/double(count_true+count_false);
}

//测试集类成员函数，统计预测正确的个数
template<class T,class T2> int Tdata<T,T2>::crector_count()
{
	int count=0;
	for(int i=0;i<num;i++)
		if(label[i][0]==label[i][1])
			count++;
	return count;
}


#endif