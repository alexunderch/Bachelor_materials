#include "ini.h"

using namespace std;

template <class K, class V>
ostream& operator << (ostream& os, const unordered_map<K, V>& m) {
  os << "{";
  bool first = true;
  for (const auto& kv : m) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << kv.first << ": " << kv.second;
  }
  return os << "}";
}


namespace Ini 
{
Section& Document::AddSection(string name){
    return sections[name];
}
const Section& Document::GetSection(const string& name) const {
    return sections.at(name);
}
size_t Document::SectionCount() const{
    return sections.size();
}

string_view Lstrip(string_view line) {
  while (!line.empty() && isspace(line[0])) {
    line.remove_prefix(1);
  }
  return line;
}

pair<string_view, string_view> Split(string_view line, char by) {
  size_t pos = line.find(by);
  string_view left = line.substr(0, pos);

  if (pos < line.size() && pos + 1 < line.size()) {
    return {left, line.substr(pos + 1)};
  } else {
    return {left, string_view()};
  }
}

Document Load(istream& input){
    string line;
    Document doc;
    Section* sec;
    string key;    
    while (getline(input, line)){
        if (!line.empty()) Lstrip(line);
        if (line[0] == '[') {
            line.erase(line.begin());
            line.erase(line.end() - 1);
            sec = &doc.AddSection(line);
            key = line;
        }     
         else if (!line.empty()) {
             auto pair_ = Split(string_view(line), '=');
             sec -> insert(pair_);
         }
    }

    return doc;
}

}  