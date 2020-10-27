#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdlib>
using namespace std;

void NextMonth(int&, vector<vector<string>>&, const vector<int>&);

int main(void)
{
	vector<vector<string>> plans_for_a_day;
	vector<int> days_of_month = {31, 28, 31, 30, 31, 30,
												31, 31, 30, 31, 30, 31};
	int Q = 0;
	cin >> Q;
	int month = 0;
	
	plans_for_a_day.resize(days_of_month[month]);
	string cmd, comment;
	while (Q)
	{
		int ind = 0;
		 	
		cin >> cmd;
		
			
		if (cmd == "ADD")
		{
			cin >> ind >> comment;
			plans_for_a_day[ind - 1].push_back(comment);
		}
		
		if (cmd == "NEXT")
		{
			NextMonth(month, plans_for_a_day, days_of_month);
		}
		
		if (cmd == "DUMP")
		{
			cin >> ind;
			cout << plans_for_a_day[ind - 1].size();
			for (auto it: plans_for_a_day[ind - 1])
				cout << " " << it;
			cout << endl; 
		}
		--Q;
	}							
	return 0;
}

void NextMonth(int& m, vector<vector<string>>& p, const vector<int>& s)
{
	m++;
	if (m > 11) m = 0;
	vector<vector<string>> tmp(p);
	int x = p.size(); 
	int y = s[m]; 
	//cout << "kjf" << "y=  " << y << endl;
	const int e = y - 1; //the last day of a new month  
	p.resize(y);
	for (int i = y; i < x; ++i)
	{
		p[e].insert(end(p[e]), begin(tmp[i]), end(tmp[i]));
	} 
	
	tmp.clear();
}
