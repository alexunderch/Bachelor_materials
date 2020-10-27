#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
using namespace std;

struct Specialization
{
  string val;
  explicit Specialization(const string& n_specialisation)
  {
    val = n_specialisation;
  }
};

struct Course
{
  string val;
  explicit Course(const string& new_course)
  {
    val = new_course;
  }
};

struct Week
{
  string val;
  explicit Week (const string& new_week)
  {
    val = new_week;
  }
};

struct LectureTitle {
  string specialization;
  string course;
  string week;
  LectureTitle(Specialization n_spec, Course n_course,
    Week n_week)
  {
    specialization = n_spec.val;
    course = n_course.val;
    week = n_week.val;
  }
};

int main()
{
  return 0;
}
