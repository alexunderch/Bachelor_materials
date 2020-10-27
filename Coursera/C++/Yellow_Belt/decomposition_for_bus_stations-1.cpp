#include <string>
#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <utility>
using namespace std;

enum class QueryType {
  NewBus,
  BusesForStop,
  StopsForBus,
  AllBuses
};

struct Query {
  QueryType type;
  string bus;
  string stop;
  vector<string> stops;
};

istream& operator >> (istream& is, Query& q) {
  string str;
  is >> str;
  if (str == "NEW_BUS")
   {
    q.type = QueryType::NewBus;
    is >> q.bus;
    int stop_count = 0;
    is >> stop_count;
    q.stops.resize(stop_count);
    for (auto& stop : q.stops)
    {
      is >> stop;
    }
    } else if (str == "BUSES_FOR_STOP")
    {
      q.type = QueryType::BusesForStop;
      is >> q.stop;
    } else if (str == "STOPS_FOR_BUS")
    {
      q.type = QueryType::StopsForBus;
      is >> q.bus;
    } else if (str == "ALL_BUSES")
    {
      q.type = QueryType::AllBuses;
    }
    return is;
}

struct BusesForStopResponse {
  vector<string> buses;
};

ostream& operator << (ostream& os, const BusesForStopResponse& r) {
  if (r.buses.empty())
  {
    os << "No stop";
  }
  else
  {
    for (const auto& bus : r.buses)
  {
      os << bus << " ";
  }
}
  return os;
}

struct StopsForBusResponse {
  string bus;
  vector<pair <string, vector<string>>> stops_for_buses;
};

ostream& operator << (ostream& os, const StopsForBusResponse& r) {
  if (r.stops_for_buses.empty()) {
    os << "No bus" << endl;
  } else {
    for (const auto& stops_n_buses : r.stops_for_buses) {
      os << "Stop " << stops_n_buses.first << ": ";
      if (stops_n_buses.second.size() == 1) {
        os << "no interchange" << endl;
      } else {
        for (const string& bus : stops_n_buses.second) {
          if (bus != r.bus) {
            os << bus << " ";
          }
        }
          os << endl;
      }

    }
  }
  return os;
}

struct AllBusesResponse {
  map <string, vector<string>> buses_to_stops;
};

ostream& operator << (ostream& os, const AllBusesResponse& r) {
  if (r.buses_to_stops.empty()) {
    os << "No buses";
  } else {
    for (const auto& bus_item : r.buses_to_stops) {
      os << "Bus " << bus_item.first << ": ";
      for (const string& stop : bus_item.second) {
        os << stop << " ";
      }
        os << endl;
    }
  }
  return os;
}

class BusManager {
public:

  void AddBus(const string& bus, const vector<string>& stops) {
    buses_to_stops.insert(make_pair(bus, stops));
    for (const auto& stop : stops) {
      stops_to_buses[stop].push_back(bus);
    }
  }

  BusesForStopResponse GetBusesForStop(const string& stop) const {
    if (stops_to_buses.count(stop) == 0) {
      return BusesForStopResponse{vector<string>()};
    } else {
      return BusesForStopResponse{stops_to_buses.at(stop)};
    }
  }


  StopsForBusResponse GetStopsForBus(const string& bus) const {
  vector<pair<string, vector<string>>> result;
   if (buses_to_stops.count(bus) > 0) {
     for (const auto& stop : buses_to_stops.at(bus)) {
       result.push_back(make_pair(stop, stops_to_buses.at(stop)));
     }
   }
   return StopsForBusResponse{bus, result};
  }

  AllBusesResponse GetAllBuses() const {
    return AllBusesResponse{buses_to_stops};
  }
private:
  map<string, vector<string>> buses_to_stops, stops_to_buses;
};

// Не меняя тела функции main, реализуйте функции и классы выше
 int main() {
    int query_count;
   Query q;

   cin >> query_count;

   BusManager bm;
   for (int i = 0; i < query_count; ++i) {
     cin >> q;
     switch (q.type) {
     case QueryType::NewBus:
       bm.AddBus(q.bus, q.stops);
       break;
     case QueryType::BusesForStop:
       cout << bm.GetBusesForStop(q.stop) << endl;
       break;
     case QueryType::StopsForBus:
       cout << bm.GetStopsForBus(q.bus) << endl;
       break;
     case QueryType::AllBuses:
       cout << bm.GetAllBuses() << endl;
       break;
     }
   }

   return 0;
 }
