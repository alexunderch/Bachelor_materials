#include "responses.h"

ostream& operator << (ostream& os, const BusesForStopResponse& r) {
  if (r.buses.empty()) {
    os << "No stop" << endl;
  } else {
    for (const auto& bus : r.buses) {
      os << bus << " ";
    }
    os << endl;
  }
  return os;
}


ostream& operator << (ostream& os, const StopsForBusResponse& r) {
  if (r.stops_for_buses.empty()) {
    os << "No bus" << endl;
  } else {
    for (const auto& stop_and_buses : r.stops_for_buses) {
      os << "Stop " << stop_and_buses.first << ":";
      if (stop_and_buses.second.size() == 1) {
        os << " no interchange" << endl;
      } else {
        for (const auto& bus : stop_and_buses.second) {
          if (bus != r.bus) {
            os << " " << bus;
          }
        }
        os << endl;
      }
    }
  }
  return os;
}


ostream& operator << (ostream& os, const AllBusesResponse& r) {
  if (r.buses_to_stops.empty()) {
    os << "No buses" << endl;
  } else {
    for (const auto& bus_and_stops : r.buses_to_stops) {
      os << "Bus " << bus_and_stops.first << ":";
      for (const auto& stop : bus_and_stops.second) {
        os << " " << stop;
      }
      os << endl;
    }
  }
  return os;
}
