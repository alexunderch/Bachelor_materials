#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;

int main() {
  int q;
  cin >> q;

  map<set<string>, int> buses;

  for (int i = 0; i < q; ++i) 
  {
    int n;
    cin >> n;
    set <string> stops;
    while (n)
    {
		string str;
		cin >> str;
		stops.insert(str);
		--n;
	}
    
    // проверяем, не существует ли уже маршрут с таким набором остановок
    if (buses.count(stops) == 0) {
      
      // если не существует, нужно сохранить новый маршрут;
      // его номер на единицу больше текущего количества маршрутов
      const int new_number = buses.size() + 1;
      buses[stops] = new_number;
      cout << "New bus " << new_number << endl;
      
    } else {
      cout << "Already exists for " << buses[stops] << endl;
    }
  }

  return 0;
}

/* Через map-y;
 * int main() {
  int q;
  cin >> q;

  map<vector<string>, int> buses;

  for (int i = 0; i < q; ++i) {
    int n;
    cin >> n;
    vector<string> stops(n);
    for (string& stop : stops) {
      cin >> stop;
    }
    
    // проверяем, не существует ли уже маршрут с таким набором остановок
    if (buses.count(stops) == 0) {
      
      // если не существует, нужно сохранить новый маршрут;
      // его номер на единицу больше текущего количества маршрутов
      const int new_number = buses.size() + 1;
      buses[stops] = new_number;
      cout << "New bus " << new_number << endl;
      
    } else {
      cout << "Already exists for " << buses[stops] << endl;
    }
  }

  return 0;
}
 * 
 * */
