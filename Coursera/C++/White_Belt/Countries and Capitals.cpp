#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <cstdlib>
using namespace std;

int main(void)
{
	int Q = 0;
	cin >> Q;
	map <string, string> COUNTRIES_w_CAPITALS;
	while (Q)
	{
		string cmd;
		cin >> cmd;
		string arg1, arg2;
		
		if (cmd == "CHANGE_CAPITAL")
		{
			cin >> arg1 >> arg2; //1-country, 2-capital;
			if (!COUNTRIES_w_CAPITALS.count(arg1))
			{
				cout << "Introduce new country " << arg1 << " with capital " << arg2 << endl;
			}
			else
			{
				const string& old_capital = COUNTRIES_w_CAPITALS[arg1]; 
				if (old_capital == arg2) 
				{
					cout << "Country " << arg1 << " hasn't changed its capital" << endl;
				}
				else 
				{
					cout << "Country " << arg1 << " has changed its capital from " 
										<< old_capital << " to " << arg2 << endl;	
				}
			}
			COUNTRIES_w_CAPITALS[arg1] = arg2;
		}
		if (cmd == "RENAME")
		{
			cin >> arg1 >> arg2;
			string old_country_name = arg1;
			string new_country_name = arg2;
			if (old_country_name == new_country_name ||
				COUNTRIES_w_CAPITALS.count(old_country_name) == 0 ||
				COUNTRIES_w_CAPITALS.count(new_country_name) == 1)
			{
				cout << "Incorrect rename, skip" << endl;
			}
			else
			{
				cout << "Country " << old_country_name << " with capital " << COUNTRIES_w_CAPITALS[old_country_name] 
				<< " has been renamed to " << new_country_name << endl;
				COUNTRIES_w_CAPITALS[new_country_name] = COUNTRIES_w_CAPITALS[old_country_name];
				COUNTRIES_w_CAPITALS.erase(old_country_name);  
			}		
		}
		if (cmd == "ABOUT")
		{
			cin >> arg1;
			string country = arg1;
			if (!COUNTRIES_w_CAPITALS.count(country))
			{
				cout << "Country " << country << " doesn't exist" << endl;
			}
			else
			{
				cout << "Country " << country << " has capital " << 
				COUNTRIES_w_CAPITALS[country] << endl;
			}
		}
		if (cmd == "DUMP")
		{
			if (COUNTRIES_w_CAPITALS.empty())
			{
				cout << "There are no countries in the world" << endl;
			}
			else 
			{
				for (const auto& item: COUNTRIES_w_CAPITALS)
				{
					cout << item.first << "/" << item.second << " ";
				}
				cout << endl;
			}
		}
	--Q;
	}

	return 0;
}
