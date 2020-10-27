#include <iomanip>
#include <iostream>
#include "profile.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>
using namespace std;

class ReadingManager {
public:
  ReadingManager()
      : pages_rating(1001),
      users_(MAX_USER_COUNT_, 0) {}

  void Read(const int& user_id, const int& page_count) {
  if (!users_[user_id]) {
      pages_rating[0].insert(user_id);
  }
  for(int i = users_[user_id] + 1; i <= page_count; ++i)
  {
    pages_rating[i].insert(user_id);
  }
  users_[user_id] = page_count;
}

  double Cheer(const int& user_id) const {

    const int user_count = pages_rating[0].size();
    if(!users_[user_id]) return 0;
    if (user_count == 1) return 1;
    return (user_count - pages_rating[users_[user_id]].size()) * 1.0 / (user_count - 1);

  }

private:
  // Статическое поле не принадлежит какому-то конкретному
  // объекту класса. По сути это глобальная переменная,
  // в данном случае константная.
  // Будь она публичной, к ней можно было бы обратиться снаружи
  // следующим образом: ReadingManager::MAX_USER_COUNT.
  static const int MAX_USER_COUNT_ = 100'001;

  vector <int> users_; // <id -> pages_count>
  vector<set<int>> pages_rating; // <set<id>>
};


int main() {
  // Для ускорения чтения данных отключается синхронизация
  // cin и cout с stdio,
  // а также выполняется отвязка cin от cout
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  {
  LOG_DURATION("sdf");


  ReadingManager manager;

  int query_count;
  cin >> query_count;

  for (int query_id = 0; query_id < query_count; ++query_id) {
    string query_type;
    cin >> query_type;
    int user_id;
    cin >> user_id;

    if (query_type == "READ") {
      int page_count;
      cin >> page_count;
      manager.Read(user_id, page_count);
    } else if (query_type == "CHEER") {
      cout << setprecision(6) << manager.Cheer(user_id) << "\n";
    }
  }
  }
  return 0;
}
