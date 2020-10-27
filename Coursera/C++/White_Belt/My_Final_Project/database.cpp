#include "database.h"

Database::Database() {};
void Database:: Add(const Date& date, const string& event)
{
  if(database[date].count(event) == 0)
  {
    database[date].insert(event);
    database_history[date].push_back(event);
  }
}

void Database::Print(ostream& out) const
{
	for (auto dat : database_history)
  {
		for (auto el : dat.second)
    {
			out << dat.first << " " << el << endl;
		}
	}
}

pair <Date, string> Database::Last(const Date& date) const
{
  auto it = database.upper_bound(date);
  if (it != database.begin())
  {
    --it;
    return make_pair(it -> first, database_history.at(it -> first).back());
  }
  else
  {
    throw invalid_argument ("Your date couldn't stand before the begin");
  }
}
