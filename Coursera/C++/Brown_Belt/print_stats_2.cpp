#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <optional>
using namespace std;

template <typename Iterator>
class IteratorRange {
public:
  IteratorRange(Iterator begin, Iterator end)
    : first(begin)
    , last(end)
  {
  }

  Iterator begin() const {
    return first;
  }

  Iterator end() const {
    return last;
  }

private:
  Iterator first, last;
};

struct Person {
  string name;
  int age, income;
  bool is_male;
};

vector<Person> ReadPeople(istream& input) {
  int count;
  input >> count;

  vector<Person> result(count);
  for (Person& p : result) {
    char gender;
    input >> p.name >> p.age >> p.income >> gender;
    p.is_male = gender == 'M';
  }

  return result;
}

template<typename Iterator>
optional<string> FindTheMostPopularName (IteratorRange<Iterator> range){
  if (range.begin() == range.end()) return nullopt;
  else{
    sort(range.begin(), range.end(), [](const Person& lhs, const Person& rhs) {
      return lhs.name < rhs.name;
    });
    const string* most_popular_name = &range.begin() -> name;
    int count = 1;
    for (auto i = range.begin(); i != range.end(); ) {
      auto same_name_end = find_if_not(i, range.end(), [i](const Person& p) {
        return p.name == i -> name;
      });
      auto cur_name_count = distance(i, same_name_end);
      if (cur_name_count > count) {
        count = cur_name_count;
        most_popular_name = &i->name;
      }
      i = same_name_end;
    }
    return *most_popular_name;
  }
}

struct Stats {
  optional<string> most_pop_f_name;
  optional<string> most_pop_m_name;
  vector<int> cumulative_wealth;
  vector<Person> sorted_by_age;
};

Stats CreateStats (vector<Person> people){
  Stats result;
  IteratorRange males {begin(people), 
                       partition(people.begin(), people.end(), [](const Person& p) {
                         return p.is_male;
                       })};
  IteratorRange females {males.end(), people.end()};

  result.most_pop_f_name = FindTheMostPopularName(females);
  result.most_pop_m_name = FindTheMostPopularName(males);

  sort(people.begin(), people.end(), [](const Person& lhs, const Person& rhs) {
    return lhs.income > rhs.income;
  });

  auto& wealth = result.cumulative_wealth;
  wealth.resize(people.size());
  if (!people.empty()) {
    wealth[0] = people[0].income;
    for (size_t i = 1; i < people.size(); ++i) {
      wealth[i] = wealth[i - 1] + people[i].income;
    }
  }
  

  sort(begin(people), end(people), [](const Person& lhs, const Person& rhs) {
    return lhs.age < rhs.age;
  });
  result.sorted_by_age = move(people);

  return result;
}

int main() {
  const Stats sts = CreateStats(ReadPeople(cin));

  for (string command; cin >> command; ) {
    if (command == "AGE") {
      int adult_age;
      cin >> adult_age;

      auto adult_begin = lower_bound(
        begin(sts.sorted_by_age), end(sts.sorted_by_age), adult_age, [](const Person& lhs, int age) {
          return lhs.age < age;
        }
      );

      cout << "There are " << std::distance(adult_begin, end(sts.sorted_by_age))
           << " adult people for maturity age " << adult_age << '\n';

    } else if (command == "WEALTHY") {
      int count;
      cin >> count;
      cout << "Top-" << count << " people have total income " << sts.cumulative_wealth[count -1] << '\n';
    } else if (command == "POPULAR_NAME") {
      char gender;
      cin >> gender;
      const auto& most_popular_name = gender == 'M' ? sts.most_pop_m_name
                                                    : sts.most_pop_f_name;
      if(most_popular_name) cout << "Most popular name among people of gender " << gender << " is " 
                                                                        << *most_popular_name << '\n';
      else cout << "No people of gender " << gender << '\n';
      }
    }
  }
