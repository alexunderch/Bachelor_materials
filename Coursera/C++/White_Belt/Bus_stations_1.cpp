#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <cstdlib>
using namespace std;

void PrintVector(const vector<string>& s)
{
	for (auto w: s)
		{
			cout << w << " ";
		}
}

int main(void)
{
	map <string, vector<string> > Bus_stations;
	vector<string> Bus_order;
	
	int Q = 0;
	cin >> Q;

	while (Q)
	{
		int stop_count = 0;
		string stop;
		string cmd, bus;
		vector<string> Stops(stop_count);
		cin >> cmd;
		
		if (cmd == "NEW_BUS")
		{
			cin >> bus >> stop_count;
			Bus_order.push_back(bus);
			
			while(stop_count)
			{
				cin >> stop;
				Stops.push_back(stop);
				--stop_count;
			}
			Bus_stations[bus] = Stops;
		}
		
		if (cmd == "BUSES_FOR_STOP")
		{
			cin >> stop;
			vector<string> tmp;
			for (auto b: Bus_order)
			{
				for (const auto& item: Bus_stations)
				{	
						if(item.first == b)
						{
							for (auto w: item.second)
							{
								if (w == stop)
								{
									tmp.push_back(b);
								}
							}
						}
				}
			}
			if (!tmp.empty())
			{
				PrintVector(tmp);
				cout << endl;
			}
			if (tmp.empty())
				{
					cout << "No stop" << endl; 
				}
			
		}
		if (cmd == "STOPS_FOR_BUS")
		{	
			vector<string> tmp;
			cin >> bus;
			if (Bus_stations.count(bus))
			{
				for (auto w: Bus_stations[bus])
				{
					cout << "Stop " << w << ": ";
					for (auto b: Bus_order)
					{
						if (b != bus)
						{
							for (auto s: Bus_stations[b])
							{
								if (s == w)
								{
									tmp.push_back(b);
								}
							}
						}					
					}
					if (tmp.empty())
					{
						cout << "no interchange" << endl;
					}
					else
					{
						PrintVector(tmp);
						cout << endl;
					}
					tmp.clear();
				}
			}		
			else
			{
				cout << "No bus" << endl;
			}
		}
		if (cmd == "ALL_BUSES")
		{
			if (!Bus_stations.empty())
			{
				for (const auto& item: Bus_stations)
				{
					cout << "Bus " << item.first << ": ";
					 PrintVector(item.second);
					 cout << endl;
				}
			}
			else 
			{
				cout << "No buses" << endl;
			}
		}
		--Q;
	}
	Bus_order.clear();
	return 0;
}



/*
 * #include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;

// ответ на запрос BUSES_FOR_STOP
void PrintBusesForStop(map<string, vector<string>>& stops_to_buses,
                       const string& stop) {
  if (stops_to_buses.count(stop) == 0) {
    cout << "No stop" << endl;
  } else {
    for (const string& bus : stops_to_buses[stop]) {
      cout << bus << " ";
    }
    cout << endl;
  }                
}

// ответ на запрос STOPS_FOR_BUS
void PrintStopsForBus(map<string, vector<string>>& buses_to_stops,
                      map<string, vector<string>>& stops_to_buses,
                      const string& bus) {
  if (buses_to_stops.count(bus) == 0) {
    cout << "No bus" << endl;
  } else {
    for (const string& stop : buses_to_stops[bus]) {
      cout << "Stop " << stop << ": ";
      
      // если через остановку проходит ровно один автобус,
      // то это наш автобус bus, следовательно, пересадки тут нет
      if (stops_to_buses[stop].size() == 1) {
        cout << "no interchange";
      } else {
        for (const string& other_bus : stops_to_buses[stop]) {
          if (bus != other_bus) {
            cout << other_bus << " ";
          }
        }
      }
      cout << endl;
    }
  }             
}

// ответ на запрос ALL_BUSES
void PrintAllBuses(const map<string, vector<string>>& buses_to_stops) {
  if (buses_to_stops.empty()) {
    cout << "No buses" << endl;
  } else {
    for (const auto& bus_item : buses_to_stops) {
      cout << "Bus " << bus_item.first << ": ";
      for (const string& stop : bus_item.second) {
        cout << stop << " ";
      }
      cout << endl;
    }
  }             
}

int main() {
  int q;
  cin >> q;

  map<string, vector<string>> buses_to_stops, stops_to_buses;

  for (int i = 0; i < q; ++i) {
    string operation_code;
    cin >> operation_code;

    if (operation_code == "NEW_BUS") {

      string bus;
      cin >> bus;
      int stop_count;
      cin >> stop_count;
      
      // с помощью ссылки дадим короткое название вектору
      // со списком остановок данного автобуса;
      // ключ bus изначально отсутствовал в словаре, поэтому он автоматически
      // добавится туда с пустым вектором в качестве значения
      vector<string>& stops = buses_to_stops[bus];
      stops.resize(stop_count);
      
      for (string& stop : stops) {
        cin >> stop;
        // не забудем обновить словарь stops_to_buses
        stops_to_buses[stop].push_back(bus);
      }

    } else if (operation_code == "BUSES_FOR_STOP") {

      string stop;
      cin >> stop;
      PrintBusesForStop(stops_to_buses, stop);

    } else if (operation_code == "STOPS_FOR_BUS") {

      string bus;
      cin >> bus;
      PrintStopsForBus(buses_to_stops, stops_to_buses, bus);

    } else if (operation_code == "ALL_BUSES") {

      PrintAllBuses(buses_to_stops);

    }
  }

  return 0;
}

 * */

