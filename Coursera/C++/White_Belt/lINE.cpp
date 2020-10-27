#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdlib>
using namespace std;

void WORRY (vector<bool>&, int);
void QUIET (vector<bool>&, int);
void COME (vector<bool>&, int);
void WORRY_COUNT (const vector<bool>&);
void PrintVector (const vector<bool>& v)
{
	for (auto i: v) 
		cout << i << " ";
	cout << endl;
}

int main(void)
{
	vector<bool> queue;
	
	int N = 0;
	cin >> N;
	string cmd;
	int ind = 0;
	
	while (N)
	{
		cin >> cmd;
		
		if (cmd == "WORRY")
		{
			cin >> ind;
			WORRY(queue, ind);
			//PrintVector(queue);
		}
		if (cmd == "QUIET")
		{
			cin >> ind;
			QUIET(queue, ind);
			//PrintVector(queue);
		}
		if (cmd == "COME")
		{
			cin >> ind;
			COME(queue, ind);
			//PrintVector(queue);
		}
		if (cmd == "WORRY_COUNT")
		{
			WORRY_COUNT(queue);
		}
		--N;
		ind = 0; 
	}
	return 0;
}

void WORRY (vector<bool>& q, int i)
{
	q[i] = true;
}

void QUIET (vector<bool>& q, int i)
{
	q[i] = false;
}

void COME (vector<bool>& q, int capacity)
{
	if (capacity >= 0)
	{
		for (int k = 0; k < capacity; ++k)
		{
			q.push_back(false);
		}
	
	}
	if (capacity < 0)
	{
		for (int k = 0; k < -capacity; ++k)
		{
			q.pop_back();
		}
	}
}

void WORRY_COUNT (const vector<bool>& q)
{
	int count = 0;
	for (auto e: q)
	{
		if (e) ++count;
	}
	cout << count << endl;
}



/*
 * #include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
  int q;
  cin >> q;
  vector<bool> is_nervous;

  for (int i = 0; i < q; ++i) {
    string operation_code;
    cin >> operation_code;

    if (operation_code == "WORRY_COUNT") {
      
      // подсчитываем количество элементов в векторе is_nervous, равных true
      cout << count(begin(is_nervous), end(is_nervous), true) << endl;
      
    } else {
      if (operation_code == "WORRY" || operation_code == "QUIET") {
        
        int person_index;
        cin >> person_index;
        
        // выражение в скобках имеет тип bool и равно true для запроса WORRY,
        // поэтому is_nervous[person_index] станет равным false или true
        // в зависимости от operation_code
        is_nervous[person_index] = (operation_code == "WORRY");
        
      } else if (operation_code == "COME") {
        
        int person_count;
        cin >> person_count;
        
        // метод resize может как уменьшать размер вектора, так и увеличивать,
        // поэтому специально рассматривать случаи с положительным
        // и отрицательным person_count не нужно
        is_nervous.resize(is_nervous.size() + person_count, false);
        
      }
    }
  }

  return 0;
}

 * 
 * 
 * 
 * */
