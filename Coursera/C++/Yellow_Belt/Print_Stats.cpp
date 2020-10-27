#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
/*
enum class Gender
{
  FEMALE,
  MALE
};

struct Person {
  int age;  // возраст
  Gender gender;  // пол
  bool is_employed;  // имеет ли работу
};

template <typename InputIt>
int ComputeMedianAge(InputIt range_begin, InputIt range_end) {
  if (range_begin == range_end) {
    return 0;
  }
  vector<typename InputIt::value_type> range_copy(range_begin, range_end);
  auto middle = begin(range_copy) + range_copy.size() / 2;
  nth_element(
      begin(range_copy), middle, end(range_copy),
      [](const Person& lhs, const Person& rhs) {
        return lhs.age < rhs.age;
      }
  );
  return middle->age;
}

void PrintStats(vector<Person> pe);

int main() {
  vector<Person> persons = {
      {31, Gender::MALE, false},
      {40, Gender::FEMALE, true},
      {24, Gender::MALE, true},
      {20, Gender::FEMALE, true},
      {80, Gender::FEMALE, false},
      {78, Gender::MALE, false},
      {10, Gender::FEMALE, false},
      {55, Gender::MALE, true},
  };
  PrintStats(persons);
  return 0;
}
*/
void PrintStats(vector<Person> pe)
{
  auto females_end = partition(begin(pe), end(pe), [](Person p)
  {
    return p.gender == Gender::FEMALE;
  });
  auto unemployed_females_end = partition(begin(pe), females_end, [](Person p)
  {
    return p.is_employed == false;
  });
  auto males_start = females_end;
  auto unemployed_males_end = partition(males_start, pe.end(), [](Person p)
  {
    return p.is_employed == false;
  });

  cout << "Median age = " << ComputeMedianAge(pe.begin(), pe.end()) << endl;
  cout << "Median age for females = " << ComputeMedianAge(pe.begin(), females_end) << endl;
  cout << "Median age for males = " << ComputeMedianAge(males_start, pe.end()) << endl;
  cout << "Median age for employed females = " << ComputeMedianAge(unemployed_females_end, females_end) << endl;
  cout << "Median age for unemployed females = " << ComputeMedianAge(begin(pe), unemployed_females_end) << endl;
  cout << "Median age for employed males = " << ComputeMedianAge(unemployed_males_end, end(pe)) << endl;
  cout << "Median age for unemployed males = " << ComputeMedianAge(males_start, unemployed_males_end) << endl;

}
