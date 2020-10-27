#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>
#include <cmath>
using namespace std;
struct FullName
{
	string FirstName;
	string LastName; 
};

class Person {
public:

  void ChangeFirstName(int year, const string& first_name) 
  {
	if ((FY.count(year) == 0))
	{
		FY[year].LastName = "";
	}
	FY[year].FirstName = first_name;
  }
  void ChangeLastName(int year, const string& last_name)
  {
	if ((FY.count(year) == 0))
		{
			FY[year].FirstName = "";
		}
	FY[year].LastName = last_name;
  }
  string GetFullName(int year)
  {
    // получить имя и фамилию по состоянию на конец года year
      if (FY.size() == 0)
      {
		  return "Incognito";
	  }
	  else
	  {
		  for (const auto& item: FY)
		  {
			  if (year < item.first)
			  {
				return "Incognito";
			  }
			  break;
		  }
	  }
	  
	  string n = "";
	  string s = "";
	  
	  for (const auto& item: FY)
	  {
		  if (item.first <= year && !item.second.FirstName.empty())
		  {
			 n =  item.second.FirstName ;
		  }
		  if (item.first <= year && !item.second.LastName.empty())
		  {
			  s = item.second.LastName;
		  }
	 }  
		  if (s == "") 
		  {
			  return n + " with unknown last name";
		  }
		  else if (n == "")
		  { 
			  return s + " with unknown first name";
		  }
		  else return n + " " + s;
}
private:
  // приватные поля
  map <int, FullName> FY;
};


int main()
{
 Person person;
  
  person.ChangeFirstName(1965, "Polina");
  person.ChangeLastName(1967, "Sergeeva");
  for (int year : {1900, 1965, 1990}) {
    cout << person.GetFullName(year) << endl;
  }
  
  person.ChangeFirstName(1970, "Appolinaria");
  for (int year : {1969, 1970}) {
    cout << person.GetFullName(year) << endl;
  }
  
  person.ChangeLastName(1968, "Volkova");
  for (int year : {1969, 1970}) {
    cout << person.GetFullName(year) << endl;
  }
  
  return 0;
}

/*
 * #include <map>
#include <string>

// если имя неизвестно, возвращает пустую строку
string FindNameByYear(const map<int, string>& names, int year) {
  string name;  // изначально имя неизвестно
  
  // перебираем всю историю по возрастанию ключа словаря, то есть в хронологическом порядке
  for (const auto& item : names) {
    // если очередной год не больше данного, обновляем имя
    if (item.first <= year) {
      name = item.second;
    } else {
      // иначе пора остановиться, так как эта запись и все последующие относятся к будущему
      break;
    }
  }
  
  return name;
}

class Person {
public:
  void ChangeFirstName(int year, const string& first_name) {
    first_names[year] = first_name;
  }
  void ChangeLastName(int year, const string& last_name) {
    last_names[year] = last_name;
  }
  string GetFullName(int year) {
    // получаем имя и фамилию по состоянию на год year
    const string first_name = FindNameByYear(first_names, year);
    const string last_name = FindNameByYear(last_names, year);
    
    // если и имя, и фамилия неизвестны
    if (first_name.empty() && last_name.empty()) {
      return "Incognito";
    
    // если неизвестно только имя
    } else if (first_name.empty()) {
      return last_name + " with unknown first name";
      
    // если неизвестна только фамилия
    } else if (last_name.empty()) {
      return first_name + " with unknown last name";
      
    // если известны и имя, и фамилия
    } else {
      return first_name + " " + last_name;
    }
  }

private:
  map<int, string> first_names;
  map<int, string> last_names;
};

 * 
 * 
 * */
