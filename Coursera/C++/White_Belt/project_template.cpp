// Реализуйте функции и методы классов и при необходимости добавьте свои
#include <map>
#include <iostream>
#include <string>
#include <set>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <exception>

using namespace std;

class Date { //Year-Month-Day
public:
  Date (int new_year, int new_month, int new_day)
  {
    year = new_year;
    month = new_month;
    day = new_day;
  }

  int GetYear() const
  {
    return year;
  }
  int GetMonth() const
  {
    return month;
  }
  int GetDay() const
  {
    return day;
  }

private:
  int year;
  int month;
  int day;
};

Date ParseDate (const string& date)
{
  istringstream date_stream(date);
  bool s = true;

  char c;

  int year = 1;
  s = s && (date_stream >> year);
  s = s && (date_stream.peek() == '-');
  date_stream.ignore(1);

  int month = 1;
  s = s && (date_stream >> month);
  s = s && (date_stream.peek() == '-');
  date_stream.ignore(1);

  int day = 1;
  s = s && (date_stream >> day);
  s = s && date_stream.eof();

  if (!s) {
    throw logic_error("Wrong date format: " + date);
    }

  if (month > 12 || month <= 0) throw logic_error(
              "Month value is invalid: " + to_string(month));
  if (day > 31 || day <= 0) throw logic_error(
              "Day value is invalid: " + to_string(day));


    return Date(year, month, day);


}

bool operator<(const Date& lhs, const Date& rhs)
{

 vector<int> a {lhs.GetYear(), lhs.GetMonth(), lhs.GetDay()};
 vector<int> b {rhs.GetYear(), rhs.GetMonth(), rhs.GetDay()};
 return a < b;
}

class Database {
public:
  void AddEvent(const Date& date, const string& event)
  {
    database[date].insert(event);
  }

  bool DeleteEvent(const Date& date, const string& event)
  {
    if (database.count(date))
    {
      if (database[date].count(event))
      {
        database[date].erase(event);

        if (database[date].size() == 0)
        {
          database.erase(date);
        }
        return true;
      }
      else return false;
    }
    else return false;
  }
  int DeleteDate(const Date& date)
  {
    int N = database[date].size();
    database.erase(date);
    return N;
  }

   set<string> Find(const Date& date) const
  {
    if (database.count(date))
    {
      return database.at(date);
    }
    return {};
  }

  void Print() const
  {
    for (const auto& item: database)
    {
      if (database.count(item.first))
      {
        for (const auto& event: database.at(item.first))
        {
          cout << setfill('0');
          cout << setw(4) << item.first.GetYear() << "-"
          << setw(2) << item.first.GetMonth() << "-" << setw(2) << item.first.GetDay();

          cout << " " << event << endl;
        }
      }
    }
  }
private:
  map < Date, set<string> > database;
};


int main() {
  try {
  Database db;

  string command;
  string data;
  string event;
  while (getline(cin, command))
  {
    istringstream ss(command);
    string com;
    ss >> com;
    if (com.size() == 0)
    {
      continue;
    }
    if (com == "Add")
    {
      ss >> data >> event;
      db.AddEvent(ParseDate(data), event);
    }
    if (com == "Del")
    {
      ss >> data;
      Date date = ParseDate(data);

      if(!ss.eof())
      {
        ss >> event;
        if (db.DeleteEvent(date, event))
        {
          cout << "Deleted successfully" << endl;
        }
        else
        {
          cout << "Event not found" << endl;
        }
      }
      else
      {
        cout << "Deleted " << db.DeleteDate(date) << " events" << endl;
      }
    }
    if (com == "Find")
    {
      ss >> data;
      Date date = ParseDate(data);
      set<string> forprint = db.Find(date);
      for (const auto& it: forprint)
      {
        cout << it << endl;
      }
    }
    if (com == "Print") db.Print();
    if (com != "Add" && com != "Del" && com != "Find" && com != "Print")
    {
      cout << "Unknown command: " << com << endl;
    }
  }


  } catch (const exception& e)
  {
    cout << e.what();
    cout << endl;
  }
  return 0;
}
