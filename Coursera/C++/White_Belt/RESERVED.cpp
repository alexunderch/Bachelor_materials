#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdlib>
using namespace std;
vector<int> Reversed(const vector<int>&);
/*
int main(void)
{
	
	const vector<int> s = {1, 2, 3, 2, 5};
	vector<int> rs = Reverse(s);
	for (auto i: rs)
	{
		cout << i << " ";
	}
	cout << endl;
	
	return 0;
}
*/
vector<int> Reversed(const vector<int>& v) {
   
  vector<int> w = v;
  for (size_t i = 0; i < v.size(); ++i)
	{
		w[i] = v[v.size() - 1 - i]; 
	}
  return w;
}
/*
 * void Reverse1(vector<int>& v)
{
	vector<int> n_v (v);
	for (size_t i = 0; i < v.size(); ++i)
	{
		v[i] = n_v[v.size() - 1 - i]; 
	}
}
 * */

 /*
  * void Reverse0(vector<int>& v) {
  int n = v.size();  // для удобства сохраним размер вектора в переменную n
  for (int i = 0; i < n / 2; ++i) {
    // элемент с индексом i симметричен элементу с индексом n - 1 - i
    // поменяем их местами с помощью временной переменной tmp
    int tmp = v[i];
    v[i] = v[n - 1 - i];
    v[n - 1 - i] = tmp;
  }
}
  * */
