#include <iostream>
#include <string>
#include <map>

using namespace std;

 string FindByYear(const map<int, string>& m, int year)
 {
   auto it = m.upper_bound(year);

   string tmp;
   if (it != m.begin()) tmp = prev(it) -> second;

   return tmp;
 }

class Person
{
public:
 void ChangeFirstName(int year, const string& first_name) {
   // добавить факт изменения имени на first_name в год year
   first_names[year] = first_name;
 }
 void ChangeLastName(int year, const string& last_name) {
   // добавить факт изменения фамилии на last_name в год year
   last_names[year] = last_name;
 }
 string GetFullName(int year) {
   // получить имя и фамилию по состоянию на конец года year
   // с помощью двоичного поиска
   const string _first_name = FindByYear(first_names, year);
   const string _last_name = FindByYear(last_names, year);

   if (_first_name.empty() && _last_name.empty()) {
     return "Incognito";

   // если неизвестно только имя
 } else if (_first_name.empty()) {
     return _last_name + " with unknown first name";

   // если неизвестна только фам3илия
 } else if (_last_name.empty()) {
     return _first_name + " with unknown last name";

   // если известны и имя, и фамилия
   } else {
     return _first_name + " " + _last_name;
   }
 }
private:
 map<int, string> first_names;
 map<int, string> last_names;
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
