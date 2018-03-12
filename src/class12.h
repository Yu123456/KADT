// Expand Decision Tree 12
// ��չ�� Expand Decision Tree 10 
// ��չ�� Expand Decision Tree 8  ֱ�Ӵ� data �ж�ȡ����
// ��д�� 2015��6��5��
// �޸��� 2015��5��29�գ����� 10-folds ������֤
// k-fold ������֤��׼ȷ�ʶ���Ϊ��the average correct value of the 10 results
//��Ҫ˼��Ϊ�����Ⱥ�������չ����ָ��
// �޸��� 2015��5��22�����磬
// 1���� sigma^2=0 ʱ��������Ӧ�ÿ����� 0.05 ���ϣ������Ϳ���ȷ������ delta/2 �ĵ�������С�� 0.05
//�޸���2015��5��27�գ����Ӿ�������ȼ������� pair<int,int> count_depth_nodes(); 

// �޸���2015��6��5�գ�����ԭʼ���ݹ�һ����(x-x_min)/(x_max-x_min)


#ifndef CLASS12_H
#define CLASS12_H

#include<iostream>
#include<fstream>      //�ļ���
#include<sstream>
#include<ctime>     // ����ʱ��
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

//���ǰ������
template<class T, class T2> class Data;
template<class T, class T2> class Node;
template<class T, class T2> class DecisionTree;
//���ǩ����ֵ��������
template<class T,class T2> class c_MuSigma2;
//�������ݼ���
template<class T,class T2> class Tdata;
//���������࣬��Ҫ�Ǳ���ʮ�۽�����֤ƽ��ֵ����׼�ƽ������ȡ�ƽ�������
class performance;

//�����Ա����

//����ʱ����
double running_time(const clock_t &s, const clock_t &e)
{
	return static_cast<double>(e-s)/CLOCKS_PER_SEC;
}

//�������ļ�������
ifstream& open_file(ifstream &in, const string &file)
{
	in.close();
	in.clear();
	in.open(file.c_str());
	return in;
}

//������ļ�������
ofstream& open_outfile(ofstream &out, const string &file)
{
	out.close();
	out.clear();
	out.open(file.c_str(), ofstream::trunc);  //���ԭ���ļ�
	return out;
}

//������ļ�������
ofstream& open_outfile2(ofstream &out, const string &file)
{
	out.close();
	out.clear();
	out.open(file.c_str(), ofstream::app);  //׷�ӵ��ļ�ĩβ
	return out;
}

//�ַ���ת������
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

//�����Ա�����������ֵ����׼��
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


//���������࣬��Ҫ�Ǳ���ʮ�۽�����֤ƽ��ֵ����׼�ƽ������ȡ�ƽ�������
class performance
{
public:
	double mean;          //��ȷ��ƽ��ֵ
	double standar_var;   //��ȷ�ʱ�׼��
	double depth;          //����ƽ�����
	double s_depth;        //���ı�׼��
	double nodes;          //����ƽ�������
	double s_nodes;        //�������׼��

	performance(){}      //Ĭ�Ϲ��캯��
	performance(double p_m, double p_v, double p_d, double p_ds,double p_n,double p_ns):mean(p_m),
		standar_var(p_v),depth(p_d),s_depth(p_ds),nodes(p_n),s_nodes(p_ns){}
};

// k-fold cross-validation ������������ȷ��ƽ��ֵ����׼�� ,������ȣ�ƽ�������
// folds ������֤������vec ���ݱ������
template<class T,class T2> performance fold_cross_validation(int folds, const vector<int> &vec,  Data<T,T2> &Da)
{
	int m_TrainData=vec.size();
	int m=m_TrainData/folds;   //ÿһ�����ݵ��������һ�ۿ��ܻ�������
	vector<double> v_rate;     //��¼ÿһ����ȷ��
	vector<double> v_depth;
	vector<double> v_nodes;
	//ǰfolds-1����֤��/���Լ�����һ����
	for(int i=0; i<folds-1; i++)
	{
		//��֤��
		vector<int> vec_test(vec.begin()+i*m,vec.begin()+(i+1)*m);
		//������Լ������Ƿ�������
		//for(vector<int>::iterator it=vec_test.begin();it!=vec_test.end();it++)
			//cout<<*it<<endl;
		//cout<<"������"<<vec_test.size()<<endl;
		//ѵ����
		vector<int> vec_train(vec.begin(),vec.begin()+i*m);
		vector<int> vec_train2(vec.begin()+(i+1)*m,vec.end());
		vec_train.insert(vec_train.end(),vec_train2.begin(),vec_train2.end());
		//����������
		DecisionTree<T,T2> DTree(vec_train,Da);
		pair<int,int> pii=DTree.count_depth_nodes();
		v_depth.push_back(pii.first);
		v_nodes.push_back(pii.second);
		//�������Լ�����
		Tdata<T,T2> Ta(vec_test.size(),Da.dimension);
		Ta.input(vec_test,Da);
		//���о���
		DTree.MakeDecision(Ta);
		v_rate.push_back(Ta.crector_rate());   //������ȷ��
		//����������
		vec_test.clear();
		vec_train.clear();
		vec_train2.clear();
	}
	// �� folds �ν�����֤
	//��֤��
	vector<int> vec_test(vec.begin()+(folds-1)*m,vec.end());
	//������Լ������Ƿ�������
	//for(vector<int>::iterator it=vec_test.begin();it!=vec_test.end();it++)
		//cout<<*it<<endl;
	//cout<<"������"<<vec_test.size()<<endl;
	//ѵ����
	vector<int> vec_train(vec.begin(),vec.begin()+(folds-1)*m);
	//����������
	DecisionTree<T,T2> DTree(vec_train,Da);
	pair<int,int> pii=DTree.count_depth_nodes();
	v_depth.push_back(pii.first);
	v_nodes.push_back(pii.second);
	//�������Լ�����
	Tdata<T,T2> Ta(vec_test.size(),Da.dimension);
	Ta.input(vec_test,Da);
	//���о���
	DTree.MakeDecision(Ta);
	v_rate.push_back(Ta.crector_rate());    //������ȷ��
	//�������
	vec_test.clear();
	vec_train.clear();
	pair<double,double> re_p=Mean_Var(v_rate);
	pair<double,double> re_p2=Mean_Var(v_depth);
	pair<double,double> re_p3=Mean_Var(v_nodes);

	return performance(re_p.first,re_p.second,re_p2.first,re_p2.second,re_p3.first,re_p3.second);
}

//�����Ա��������
//�ݹ鹹������������
// remain_point ���ݼ���ʣ����������
// remain_attribute ���ݼ���ʣ���������ԣ���������
// da �������ݼ�
template<class T,class T2> Node<T,T2>* BuildDecisionTree(Node<T,T2> *, vector<int> ,vector<string> , Data<T,T2> & );

// T ��������ֵ���ͣ�T2 �������ǩ����
template<class T, class T2> class Data
{
public:
	int num;      //����������
	int dimension;    //����ά��/���Ը���(��������)
	T **data;     //�����ݣ���������
	T2 *label;     //���������ţ��� data �ж�Ӧ
	string *dim_label;   //���Ա�ǩ���洢������������,���һ��Ϊ���ǩ
	T *max_T;       //�������Ե����ֵ�����ڹ�һ������    // �� input1 �б����׼��
	T *min_T;        //�������Ե���Сֵ�����ڹ�һ������    // �� input1 �б����ֵ

	Data(int p_num, int p_dimension):num(p_num),dimension(p_dimension)   //���캯��
	{
		data=new T*[num];     //��̬�����ڴ�ռ�
		for(int i=0; i<num; i++)
			data[i]=new T[dimension];
		dim_label=new string[dimension];
		label=new T2[num];
		max_T=new T[dimension];
		min_T=new T[dimension];
	}
	~Data()   //��������
	{
		for(int i=0; i<num; i++)
			delete [] data[i];
		delete [] data;
		delete [] dim_label;
		delete [] label;
		delete [] max_T;
		delete [] min_T;
	}
	void input(string, vector<int>&);    //���ݶ��뺯��, ��һ�� (x-x_min)/(max-min)
	void input1(string, vector<int>&);  //���ݶ��뺯������̬��һ��  (x-mean)/var
	void print();          //������Ժ���
	bool IsAllTheSameLabel(const vector<int>&);  //�ж��Ƿ�����ͬһ��
	//����������ţ����ظ����������ǩ
	T2 return_label(int i)
	{
		return label[i];
	}
	//���ض������ǩ
	T2 MostCommonLabel(const vector<int>&);
	//������������
	string Attribute_selection_method(const vector<int>& ,const vector<string>& );
	//�����β���������Ӧ�ı��
	int return_dim_label(const string&);
	//�������Ա�����������ֵ�������
	pair<T,T> MS_f(const vector<int> &, const int );
	//�������Ա��������sigma2=0.0ʱ��ȡ�����ȴ��ڵ��� 0.95 ʱ�� sigma2
	T min_sigma2(const T2 &,const vector<int> &, const int &,const int &);
	//�������Ա���������� Expand Gini Index
	double Expand_Gini_Index(const vector<int> &, const vector<c_MuSigma2<T,T2> > &,const int &);
	//���� Gini Index
	double Gini(const vector<int>&);
};

//�����࣬���ݶ��뺯��, ivec �������洢���ݵ��ţ��� 0 ��ʼ���㣬���������Ӧ
template<class T,class T2> void Data<T,T2>::input(string filename, vector<int> &ivec)
{
	ifstream in;
	open_file(in, filename);  //���ļ�
	//���ݵ�һ��Ϊ������
	string in_str;
	getline(in, in_str);   //�����һ��
	string_replace(in_str,",","\t");   // ����ת���� "\t"
	istringstream str(in_str);   //�ַ�����
	string out_str;
	for(int i=0; i<dimension; i++)   //���ǩ
	{
		str>>out_str;
		dim_label[i]=out_str;
	}
	//�������ݴ洢�������У��漰����ָ�꣬���� for ѭ��
	T d_T;     // ���ݱ���
	T2 d_T2;    //���ű���
	bool b_flag=true;     //��ǵ�һ�����ݣ����ȱ��浽 max_T ��
	for(int j=0; j<num; j++) //��ʼ��������
	{
		getline(in, in_str);
		string_replace(in_str,",","\t");   // ����ת���� "\t"
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
				if(d_T>max_T[i])   //�������ֵ
					max_T[i]=d_T;
				if(d_T<min_T[i])   //������Сֵ
					min_T[i]=d_T;
			}
		}
		str>>d_T2;
		label[j]=d_T2;
		ivec.push_back(j);
	}
	//���й�һ��
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


//�����࣬���ݶ��뺯��, ivec �������洢���ݵ��ţ��� 0 ��ʼ���㣬���������Ӧ
//ע�⣬�˴��� min_T �����ֵ��max_T �����׼��
template<class T,class T2> void Data<T,T2>::input1(string filename, vector<int> &ivec)
{
	ifstream in;
	open_file(in, filename);  //���ļ�
	//���ݵ�һ��Ϊ������
	string in_str;
	getline(in, in_str);   //�����һ��
	string_replace(in_str,",","\t");   // ����ת���� "\t"
	istringstream str(in_str);   //�ַ�����
	string out_str;
	for(int i=0; i<dimension; i++)   //���ǩ
	{
		str>>out_str;
		dim_label[i]=out_str;
	}
	//�������ݴ洢�������У��漰����ָ�꣬���� for ѭ��
	T d_T;     // ���ݱ���
	T2 d_T2;    //���ű���
	for(int i=0;i<dimension;i++)
	{
		min_T[i]=0.0;
		max_T[i]=0.0;
	}
	for(int j=0; j<num; j++) //��ʼ��������
	{
		getline(in, in_str);
		string_replace(in_str,",","\t");   // ����ת���� "\t"
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
	//�����ֵ����׼��
	for(int i=0; i<dimension; i++)
	{
		min_T[i]=min_T[i]/double(num);                              // ��ֵ
		max_T[i]=sqrt(max_T[i]/double(num)-min_T[i]*min_T[i]);    // ��׼��
	}
	//��̬��׼�� 
	for(int j=0; j<num; j++)
	{
		for(int i=0; i<dimension; i++)
			data[j][i]=(data[j][i]-min_T[i])/max_T[i];
	}
}


//������������Ժ���
template<class T, class T2> void Data<T,T2>::print()
{
	cout<<"��������"<<endl;
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
	cout<<"���ֵΪ��"<<endl;
	for(int i=0; i<dimension; i++)
		cout<<max_T[i]<<"\t";
	cout<<endl;
	cout<<"��СֵΪ��"<<endl;
	for(int i=0; i<dimension; i++)
		cout<<min_T[i]<<"\t";
	cout<<endl;
}

//�������Ա����
//�ж������Ƿ�����ͬһ�࣬��ͬ���� true, ���򷵻� false
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

//���ݵ����Ա���������ض������ǩ
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

//���ݵ����Ա������������������
// ivec ���ݵ��ţ���Ӧ�������еı�ţ�svec ���Լ�
template<class T,class T2>	string Data<T,T2>::Attribute_selection_method(const vector<int> &ivec ,const vector<string> &svec )
{
	if(svec.size()==1)
		return svec[0];
	else
	{
		map<double,string> sd_map;   // map< Expand Gini Index,attribute> �������������� map �İ��ռ�ֵ���򱣴��ֱ��ȡ��һ������
		multimap<T2,int> ti_map;     // �����ݰ���ֿ����������ݱ��
		set<T2> t_set;         //�������ǩ
		typedef multimap<T2,int>::size_type sz_type;   
		for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
		{
			T2 label(label[*it]);
			ti_map.insert(make_pair(label, *it));
			t_set.insert(label);
		}
		for(vector<string>::const_iterator iter=svec.begin(); iter!=svec.end(); iter++)  //����ÿһ������
		{
			string s_attribute=(*iter);    //����
			int dim_i=return_dim_label(s_attribute);   //���ظ����������ݼ� data ������Ӧ�ı��
			vector<int> new_ivec;       //�洢ÿһ�����µ����ݱ��
			vector<c_MuSigma2<T,T2> > cvec;   // �洢���ǩ����ֵ������
			for(set<T2>::iterator it=t_set.begin(); it!=t_set.end();it++)   //����ÿһ����
			{
				T2 t_label=(*it);  //���ǩ
				//�����ݱ�Ű������ǩ t_label �洢�� new_ivec �У����ڼ��� mu, sigma2
				sz_type entries=ti_map.count(t_label);    // t_label ���ĸ���
				multimap<T2,int>::iterator f_it=ti_map.find(t_label);   // t_label ����ʼ������
				for(sz_type i=0; i!=entries; i++, f_it++)  // t_label ������ݱ��浽 new_ivec ��
					new_ivec.push_back(f_it->second);
				pair<T,T> MuSigma2=MS_f(new_ivec,dim_i);
				if(abs(MuSigma2.second-0.0)<1e-4)
					MuSigma2.second=min_sigma2(t_label,ivec,dim_i,new_ivec[0]);// ���ú���������̾��룬���� sigma2
				cvec.push_back(c_MuSigma2<T,T2>(t_label,MuSigma2.first,MuSigma2.second));   //�洢
				new_ivec.clear();   //��� new_ivec ���洢��һ��������ݱ��
			}
			//������չ����ָ�� Expand Gini Index
			double EGI=Expand_Gini_Index(ivec,cvec,dim_i);
			sd_map.insert(make_pair(EGI,s_attribute));
			cvec.clear();   //��������Ȳ�������
		}
		return sd_map.begin()->second;
	}
}


//�����β���������Ӧ�ı��
template<class T,class T2> int Data<T,T2>::return_dim_label(const string &str)
{
	for(int i=0; i<dimension; i++)
	{
		if(str==dim_label[i])
			return i;
	}
	return -1;
}

//�������Ա�����������ֵ�������
// ivec ���ݱ�ţ� dim ���Ա��
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

//�������Ա��������sigma2=0.0ʱ��ȡ�����ȴ��ڵ��� 0.95 ʱ�� sigma2
// t_label ��ǰ�����ǩ�� ivec ���ݼ���ţ�dim ���Ա�ţ�kk ��ǰ���һ�������������
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
		//return 2.4369657*dis*dis;   // ��ʱ delta/2 �����ȿ����� 0.95
	   // return 1.186402698*dis*dis;  //��ʱ delta/2 �����ȿ����� 0.9
	   return 0.041726*dis*dis;      //��ʱ delta/2 �����ȿ����� 0.05
		//return 0.166904*dis*dis;       // ��ʱ delta �����ȿ����� 0.05
	}
}

//�������Ա���������� Gini Index
template<class T,class T2> double Data<T,T2>::Gini(const vector<int> &ivec)
{
	if(ivec.empty())
		return 0.0;
	typedef map<T2,int> map_s_i;   // <���ǩ��int>
	map_s_i smap;
	for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
		++smap[label[*it]];
	double value=0.0;
	for(map_s_i::iterator it=smap.begin(); it!=smap.end(); it++)
		value+=double(it->second)*double(it->second);
	return 1.0-value/(double(ivec.size())*double(ivec.size()));
}

//�������Ա���������� Expand Gini Index
// ivec ���ݱ�ţ�cvec ��������Ȳ���, dim ���Ա��
template<class T,class T2> double Data<T,T2>::Expand_Gini_Index(const vector<int> &ivec, const vector<c_MuSigma2<T,T2> > &cvec,const int &dim)
{
	int m=cvec.size();    //��ĸ���
	vector<int> *p=new vector<int>[m];   //�������飬�����˳���� vector<c_MuSigma2<T,T2> > ��Ӧ�ϣ��������������ʱ�����ݱ��
	//�������������ԭ�򣬽����ݱ�ű����� p ������
	for(vector<int>::const_iterator it=ivec.begin(); it!=ivec.end(); it++)
	{
		int k=(*it);
		T var=data[k][dim];
		double best_menbership_degree=cvec[0].Menbership(var);//����������
		int best_flag=0;        //�����Ⱥ�����ǣ���Ǹ������Ⱥ����� cvec �еı��
		for(unsigned int i=1; i<cvec.size(); i++)
		{
			double bmd=cvec[i].Menbership(var);
			if(bmd>best_menbership_degree)
			{
				best_menbership_degree=bmd;
				best_flag=i;
			}
		}
		//�������ݱ��
		p[best_flag].push_back(k);
	}
	//���� Expand Gini Index
	double egi=0.0;
	for(int i=0; i<m; i++)
	{
		egi+=double(p[i].size())*Gini(p[i]);
	}
	delete [] p;   //�ͷŶ�̬�ռ�
	return egi/double(ivec.size());
}


//���ǩ����ֵ��������
template<class T,class T2> class c_MuSigma2
{
public:
	T2 class_label;
	T mu;
	T sigma2;

	c_MuSigma2(){}    //Ĭ�Ϲ��캯��
	c_MuSigma2(T2 c, T m,T s):class_label(c),mu(m),sigma2(s){}    //���캯��
	const double Menbership(const T &) const;    //�����ȼ��㺯��, const �������
	double Menbership(const T &);    //���������ȼ��㺯��
};

//��ֵ���������Ա����������������
template<class T,class T2> const double c_MuSigma2<T,T2>::Menbership(const T &var) const
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}

//��ֵ���������Ա����������������
template<class T,class T2> double c_MuSigma2<T,T2>::Menbership(const T &var)
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}


//�����
// T ����ֵ����
// T2 ���ǩ����
template<class T, class T2> class Node
{
public:
	string attribute;  //����ֵ
	T mu;    //��˹�����Ⱥ����ľ�ֵ
	T sigma2;  //��˹�����Ⱥ����ķ���  sigma^2
	T2 class_label;   //���ǩ
	vector<Node<T,T2>* > childs;   //���к���

	Node(){}   //Ĭ�Ϲ��캯��
	bool IsLeaf()   //�ж��Ƿ�Ϊ��Ҷ
	{
		if(childs.empty())
			return true;
		else
			return false;
	}
	//������Ա����������������
	double Menbership(T );
};

//������Ա����������������
template<class T,class T2> double Node<T,T2>::Menbership(T var)
{
	return exp(-(var-mu)*(var-mu)/(2.0*sigma2));
}


//��������
// T ����ֵ����
// T2 ���ǩ����
template<class T, class T2> class DecisionTree
{
public:
	Node<T,T2> *root;

	//��Ա����
	//DecisionTree():root(NULL){}    //Ĭ�Ϲ��캯��
	DecisionTree(vector<int> ,Data<T,T2> &);       // �����ݼ�����������
	void FreeTree(Node<T,T2> *p)
	{
		if(p != NULL )
		{
			for(vector<Node<T,T2>* >::iterator it=p->childs.begin(); it!=p->childs.end(); it++)
				FreeTree(*it);
			delete p;
		}
	}
	// ��������
	~DecisionTree(){ FreeTree(root); }
	//���������
	void PrintDT();
	//������ļ���
	void PrintDT(ofstream &,const string &);
	//���ߺ���
	void MakeDecision(Tdata<T,T2> &);
	T2 recursion_MD(Node<T,T2> *, Tdata<T,T2> &, int);  //�����ߺ����ݹ���þ���
	pair<int,int> count_depth_nodes();        //���� Decision Tree ��ȼ�������  pair<depth,nodes>
};

//������ļ���
template<class T,class T2> void DecisionTree<T,T2>::PrintDT(ofstream &fout, const string &str)
{
	queue<Node<T,T2>* > que;
	Node<T,T2> *pointer=root;
	if(pointer)
		que.push(pointer);
	int current_n=1, next_n=0, level_n=0;
	fout<<"Expand Decision Tree"<<endl;
	fout<<" �� "<<level_n<<" ������"<<endl;
	pointer=que.front();
	que.pop();
	fout<<"�м��㣺attribute: "<<pointer->attribute<<" , ������"<<pointer->childs.size()<<endl;
	for(vector<Node<T,T2>* >::iterator it=pointer->childs.begin(); it!=pointer->childs.end(); it++)
	{
		que.push(*it);
		next_n++;
	}
	current_n=next_n;
	next_n=0;
	fout<<endl;
	level_n++;
	fout<<" �� "<<level_n<<" ������"<<endl;
	while(!que.empty())
	{
		pointer=que.front();
		que.pop();
		if(pointer->IsLeaf())
			fout<<"Ҷ��㣺 class label: "<<pointer->class_label<<" , mu= "<<pointer->mu<<" , sigma2= "<<pointer->sigma2<<"\t";
		else
			fout<<"�м��㣺attribute: "<<pointer->attribute<<" , mu= "<<pointer->mu<<" , sigma2= "<<pointer->sigma2<<" ,������"<<pointer->childs.size()<<"\t";
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
				fout<<" �� "<<level_n<<" ������"<<endl;
		}
	}
	fout<<endl;
}

//���ߺ���
template<class T,class T2> void DecisionTree<T,T2>::MakeDecision(Tdata<T,T2> &td)
{
	Node<T,T2> *p=root;
	//��ȡ���Լ������������������һ�µĶ�Ӧ����
	int t_num=td.num;
	for(int i=0; i<t_num; i++)
		td.label[i][1]=recursion_MD(root,td,i);
}

//�����ߺ����ݹ���þ���
template<class T,class T2> T2 DecisionTree<T,T2>::recursion_MD(Node<T,T2> *p, Tdata<T,T2> &td, int i)
{
	if(p->IsLeaf())
		return p->class_label;
	else
	{
		string attribute=p->attribute;    //��ȡ��������
		int i_dim=td.re_dim_label(p->attribute);  //������Ҫ���Ե����Ե�ά��ֵ
		T var=td.data[i][i_dim];    //����ֵ

		//ѡȡ�������������
		double menbership_degree=p->childs[0]->Menbership(var);
		unsigned int flag=0;    //���ʹ�õڼ��������Ⱥ�������ѡ����һ������
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



//���캯��
// remain_point ���ݼ���ʣ����������
// remain_attribute ���ݼ���ʣ���������ԣ���������
// da �������ݼ����������� remain_point ��ı�Ż�ȡ����������
template<class T, class T2> DecisionTree<T,T2>::DecisionTree(vector<int> remain_point,Data<T,T2> &da )
{
	root=new Node<T,T2>();
	// �տ�ʼ��Ҫ������������
	vector<string> remain_attribute(da.dim_label, da.dim_label+da.dimension);   
	root=BuildDecisionTree(root, remain_point, remain_attribute, da);
}

//�ݹ鹹������������
// remain_point ���ݼ���ʣ����������
// remain_attribute ���ݼ���ʣ���������ԣ���������
// da �������ݼ�
template<class T,class T2> Node<T,T2>* BuildDecisionTree(Node<T,T2> *p, vector<int> remain_point,
											vector<string> remain_attribute, Data<T,T2> &da )
{
	if(p==NULL)      //��� Node ָ����Ϊ�գ��򴴽�һ��
		p=new Node<T,T2>();
	//��� remain_point �����е�ͬһ�࣬��ʱ Node ΪҶ��㣬�������
	if(da.IsAllTheSameLabel(remain_point))
	{
		//�������ǩ
		p->class_label=da.return_label(remain_point[0]);
		return p;
	}
	//��� remain_attribute ��������Ѿ�������ϣ�����δ����������㣬�ö������ǩ��Ǹ���
	if(remain_attribute.empty())
	{
		//���ض������ǩ
		p->class_label=da.MostCommonLabel(remain_point);
		return p;
	}
	//�ҳ���õ�����
	string best_attribute=da.Attribute_selection_method(remain_point,remain_attribute);
	// �� best_attribute ��ǽ��
	p->attribute=best_attribute;
	//�������Լ���ɾ�� best_attribute ����ֵ
	vector<string> new_remain_attribute;
	for(vector<string>::iterator it=remain_attribute.begin(); it!=remain_attribute.end(); it++)
	{
		if(*it !=best_attribute)
			new_remain_attribute.push_back(*it);
	}
	//���� best_attribute �µķ���׼�򣬼�ÿ����������Ⱥ���
	multimap<T2,int> ti_map;     // �����ݰ���ֿ����������ݱ��
	set<T2> t_set;         //�������ǩ
	typedef multimap<T2,int>::size_type sz_type;   
	for(vector<int>::iterator it=remain_point.begin(); it!=remain_point.end(); it++)
	{
		T2 label(da.label[*it]);
		ti_map.insert(make_pair(label, *it));
		t_set.insert(label);
	}
	//�������������µľ�ֵ������
	int best_dim=da.return_dim_label(best_attribute);   //�������Զ�Ӧ���
	vector<int> new_ivec;   //����ÿ�� splitting_criterion (��) �µ����ݵ���
	vector<c_MuSigma2<T,T2> > cvec;   //�������ǩ����ֵ������� splitting_criterion
	for(set<T2>::iterator it=t_set.begin();it !=t_set.end(); it++)
	{
		T2 t_label=(*it);  //���ǩ
		//�����ݱ�Ű������ǩ t_label �洢�� new_ivec �У����ڼ��� mu, sigma2
		sz_type entries=ti_map.count(t_label);   // t_label ���ĸ���
		multimap<T2,int>::iterator f_it=ti_map.find(t_label);
		// t_label ������ݱ��浽 new_ivec ��
		for(sz_type i=0; i!=entries; i++, f_it++)
			new_ivec.push_back(f_it->second);
		//�����ֵ������
		pair<T,T> MuSigma2=da.MS_f(new_ivec,best_dim);
		if(abs(MuSigma2.second-0.0)<1e-4)  //sigma2=0 ʱ�����¸�ֵ sigma2
			MuSigma2.second=da.min_sigma2(t_label,remain_point,best_dim,new_ivec[0]);
		cvec.push_back(c_MuSigma2<T,T2>(t_label,MuSigma2.first,MuSigma2.second));  //������������Ⱥ���
		new_ivec.clear();  // ��� new_ivec, �洢��һ��������ݱ��
	}
	//�������������ԭ�򣬻�������
	int cm=cvec.size();   //��ĸ�����
	vector<int> *data_p=new vector<int>[cm]; //�������飬�����˳���� vector<c_MuSigma2<T,T2> > ��Ӧ�ϣ��������������ʱ�����ݱ��
	//�������������ԭ�򣬽����ݱ�ű����� data_p ������
	for(vector<int>::iterator it=remain_point.begin(); it!=remain_point.end(); it++)
	{
		int k=(*it);
		T var=da.data[k][best_dim];
		double best_menbership_degree=cvec[0].Menbership(var);  //����������
		int best_flag=0;   //�����Ⱥ�����ǣ���Ǹú����� cvec �еı��
		for(unsigned int i=1; i<cvec.size(); i++)
		{
			double bmd=cvec[i].Menbership(var);
			if(bmd>best_menbership_degree)
			{
				best_menbership_degree=bmd;
				best_flag=i;
			}
		}
		//�������ݱ��
		data_p[best_flag].push_back(k);
	}
	//������֧
	for(int i=0; i<cm; i++)
	{
		Node<T,T2> *new_node=new Node<T,T2>();
		new_node->mu=cvec[i].mu;
		new_node->sigma2=cvec[i].sigma2;
		if(data_p[i].empty())   //��ǰ��֧û�����������������Ⱥ�����û�����ݵ�
		{
			//��������Ϊ������
			new_node->class_label=cvec[i].class_label;
			//���ض������ǩ
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

//��������Ա�������������������ȼ�������, pair<depth,nodes>
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
			total_n++;   //�ܽ����ͳ��
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


//�������ݼ���
template<class T,class T2> class Tdata
{
public:
	int num;  //������������
	int dimension;  //����ά����������ǩ
	T **data;  //������
	T2 **label;  //�����ݱ�ǩ���ڶ��б���Ԥ�����ǩ
	string *dim_label;  //���Ա�ǩ���������ǩ

	Tdata(int,int);   //���캯��
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
	void input(const string&);  //������Լ�
	void input(const vector<int>&, const Data<T,T2>&);   //�������л�ȡ���ݱ�Ŷ�ȡ����
	void Print();  //��ӡ���Լ�
	void Print2();   //��ӡԤ�⼯
	void Print2(ofstream &, const string&);  //��ӡԤ�⼯���ļ���
	int re_dim_label(const string &);     //���ض�Ӧ�����µ�ά��
	double crector_rate();       //����Ԥ����ȷ��
	int crector_count();         //ͳ����֤��ȷ���ݸ���
};

//���캯��
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

//Tdata ��Ա������������Լ�
template<class T,class T2> void Tdata<T,T2>::input(const string &str)
{
	ifstream in;
	open_file(in,str);
	//���ݵ�һ��Ϊ����
	string in_str;  
	getline(in,in_str);  //�����һ��
	istringstream istr(in_str);   //�ַ�����
	string out_str;
	for(int i=0; i<dimension; i++)
	{
		istr>>out_str;
		dim_label[i]=out_str;
	}
	T d_T;   //���ݱ���
	T2 d_T2;  //���ǩ����
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

//�������л�ȡ���ݱ�Ŷ�ȡ����
template<class T,class T2> void Tdata<T,T2>::input(const vector<int> &vec, const Data<T,T2> &da)
{
	for(int i=0; i<dimension; i++)   //����
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

//��ӡ���Լ�
template<class T,class T2> void Tdata<T,T2>::Print()
{
	cout<<"������Լ���"<<endl;
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

//��ӡԤ�⼯
template<class T,class T2> void Tdata<T,T2>::Print2()
{
	cout<<"���Ԥ������"<<endl;
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

//��ӡԤ�⼯���ļ���
template<class T,class T2> void Tdata<T,T2>::Print2(ofstream &fout,const string &str)
{
	fout<<endl;
	fout<<"���Ԥ������"<<endl;
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

//���ض�Ӧ�����µ�ά��
template<class T,class T2> int Tdata<T,T2>::re_dim_label(const string &str)
{
	for(int i=0; i<dimension; i++)
		if(str==dim_label[i])
			return i;
	return -1;
}

//������ȷ��
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

//���Լ����Ա������ͳ��Ԥ����ȷ�ĸ���
template<class T,class T2> int Tdata<T,T2>::crector_count()
{
	int count=0;
	for(int i=0;i<num;i++)
		if(label[i][0]==label[i][1])
			count++;
	return count;
}


#endif