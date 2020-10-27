#pragma once
#include "date.h"
#include <utility>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

template <typename t1, typename t2>
ostream& operator << (ostream& stream, const pair<t1, t2>& p)
{
  stream << p.first << " " << p.second;
  return stream;
}

class Database
{
public:
  Database();
  void Add(const Date& date, const string& event);
  void Print(ostream& out) const;
  pair <Date, string> Last(const Date& date) const;

  template<typename func>
  int RemoveIf(func pred)
  {
    int number = 0;
    map<Date, vector<string>::iterator> tmp;
    for (auto item: database_history)
    {
      auto losers = stable_partition(begin(database_history[item.first]),
      end(database_history[item.first]), [item, pred](string event)
      {
        return !pred(item.first, event);
      });
      for (auto er_ind = losers; er_ind != end(database_history[item.first]); ++er_ind)
      {
        database[item.first].erase(*er_ind);
        ++number;
      }
      if (database[item.first].size() == 0)
      {
        database.erase(item.first);
      }
      tmp[item.first] = losers;
    }
    for (auto x: tmp)
    {
      database_history[x.first].erase(x.second, database_history[x.first].end());
			if (!database_history[x.first].size())
      {
				database_history.erase(x.first);
      }
    }
    return number;
  }

template<typename func>
vector<pair<Date, string>> FindIf(const func& pred) const
  {
    vector<pair<Date, string>> result;

    for (auto item: database_history)
    {
      for (auto hist: item.second)
      {
        if(pred(item.first, hist))
          result.push_back(make_pair(item.first, hist));
      }
    }
    return result;
    }

private:
  map<Date, set<string>> database;
  map<Date, vector<string>> database_history;
};
