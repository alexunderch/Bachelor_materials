#include <iostream>
#include <iomanip>
#include <vector>
#include "date.h"

using namespace std;
Date:: Date(const int& new_year, const int& new_month, const int& new_day)
{
  year = new_year;
  month = new_month;
  day = new_day;
}

int Date::GetYear() const
{
  return year;
}

int Date::GetMonth() const
{
  return month;
}

int Date::GetDay() const
{
  return day;
}

bool operator<(const Date& lhs, const Date& rhs)
{
  return ((lhs.GetYear() < rhs.GetYear()) || (lhs.GetYear() == rhs.GetYear() && lhs.GetMonth() < rhs.GetMonth()) || (lhs.GetYear() == rhs.GetYear()
          && lhs.GetMonth() == rhs.GetMonth() && lhs.GetDay() < rhs.GetDay()));
}

bool operator == (const Date& lhs, const Date& rhs)
{
  return (lhs.GetYear() == rhs.GetYear() && lhs.GetMonth() == rhs.GetMonth() && lhs.GetDay() == rhs.GetDay());
}

bool operator <= (const Date& lhs, const Date& rhs)
{
 return (lhs < rhs) || (lhs == rhs);
}

bool operator != (const Date& lhs, const Date& rhs)
{
 return !(lhs == rhs);
}

bool operator > (const Date& lhs, const Date& rhs)
{
 return !(lhs <= rhs);
}

bool operator >= (const Date& lhs, const Date& rhs)
{
 return !(lhs < rhs);
}

ostream& operator << (ostream& stream, const Date& date)
{
  stream << setw(4) << setfill('0') << date.GetYear() << "-"
		<< setw(2) << setfill('0') << date.GetMonth() << "-"
		<< setw(2) << setfill('0') << date.GetDay();
  return stream;
}
