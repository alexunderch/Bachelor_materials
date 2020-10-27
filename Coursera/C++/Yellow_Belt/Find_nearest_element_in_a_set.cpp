#include <iostream>
#include <set>
#include <iterator>

using namespace std;

set<int>::const_iterator FindNearestElement(const set<int>& numbers, int border)
{
    const auto less_it = numbers.lower_bound(border);
    if(less_it == begin(numbers)) return less_it;
    const auto last_less_it = prev(less_it);
    if (less_it == end(numbers)) return last_less_it;

    if(border - *less_it <= *last_less_it - border)
      return last_less_it;
    else return less_it;
}

int main() {
  set<int> numbers = {1, 4, 6};
  cout <<
      *FindNearestElement(numbers, 0) << " " <<
      *FindNearestElement(numbers, 3) << " " <<
      *FindNearestElement(numbers, 5) << " " <<
      *FindNearestElement(numbers, 6) << " " <<
      *FindNearestElement(numbers, 100) << endl;

  set<int> empty_set;

  cout << (FindNearestElement(empty_set, 8) == end(empty_set)) << endl;
  return 0;
}
