#include <sstream>
#include <exception>
#include <iostream>
#include "phone_number.h"
using namespace std;

string PhoneNumber::GetInternationalNumber() const
{
  return "+" + country_code_ + "-" + city_code_ + "-" + local_number_;
}

string PhoneNumber::GetCountryCode() const
{
  return country_code_;
}

string PhoneNumber::GetCityCode() const
{
  return city_code_;
}

string PhoneNumber::GetLocalNumber() const
{
  return local_number_;
}



PhoneNumber::PhoneNumber(const string &international_number)
{
  char init = international_number[0];
  istringstream iss(international_number);
  iss.ignore(1);
  getline(iss, country_code_, '-');
  getline(iss, city_code_, '-');
  getline(iss, local_number_);

  if(init != '+' || country_code_.empty() || city_code_.empty() || local_number_.empty())
    throw invalid_argument("invalid_arguments");
}
