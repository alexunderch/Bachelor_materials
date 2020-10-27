#pragma once
#include <iostream>
#include <exception>
#include <sstream>
#include "date.h"
using namespace std;

class Date { //Year-Month-Day
public:
  Date (const int& new_year, const int& new_month, const int& new_day);

  int GetYear() const;
  int GetMonth() const;
  int GetDay() const;

private:
  int year;
  int month;
  int day;
};

ostream& operator << (ostream& stream, const Date& date);
bool operator < (const Date& lhs, const Date& rhs);
bool operator <= (const Date& lhs, const Date& rhs);
bool operator >= (const Date& lhs, const Date& rhs);
bool operator > (const Date& lhs, const Date& rhs);
bool operator == (const Date& lhs, const Date& rhs);
bool operator != (const Date& lhs, const Date& rhs);

template <typename T>
Date ParseDate (T& date_stream)
{
  bool s = true;

  int year = 0;
  s = s && (date_stream >> year);
  s = s && (date_stream.peek() == '-');
  date_stream.ignore(1);

  int month = 0;
  s = s && (date_stream >> month);
  s = s && (date_stream.peek() == '-');
  date_stream.ignore(1);

  int day = 1;
  s = s && (date_stream >> day);
  s = s && (date_stream.eof() || date_stream.peek() == ' ');

  if (!s) {
    throw logic_error("Wrong date format!");
    }

  if (year > 9999 || year <= 0) throw logic_error(
              "Year value is invalid: " + to_string(year));
  if (month > 12 || month <= 0) throw logic_error(
              "Month value is invalid: " + to_string(month));
  if (day > 31 || day <= 0) throw logic_error(
              "Day value is invalid: " + to_string(day));


    return Date(year, month, day);

}
