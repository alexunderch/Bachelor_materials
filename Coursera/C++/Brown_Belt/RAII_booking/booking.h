#pragma once
#include <utility>

namespace RAII {
template <typename Provider>
class Booking {
private:
    Provider* provider = nullptr;
public:
 Booking(Provider* p, int& count) : provider(std::move(p)) {}
 Booking (const Booking& other) = delete;

 Booking (Booking&& other) {
     provider = std::move(other.provider);
     other.provider = nullptr;
 }
 Booking& operator = (const Booking& other) = delete;
 Booking& operator = (Booking&& other) {
     if (provider) provider -> CancelOrComplete(*this);
     this -> provider = std::move(other.provider);
     other.provider = nullptr;
     return *this;

 }
 ~Booking() {
     if (provider) provider -> CancelOrComplete(*this);
 }
};
}
 /*
 #pragma once

#include <utility>

namespace RAII {

  template <typename Provider>
  class Booking {
  private:
    using BookingId = typename Provider::BookingId;

    Provider* provider;
    BookingId booking_id;

  public:
    Booking(Provider* p, const BookingId& id)
      : provider(p),
        booking_id(id)
    {
    }

    Booking(const Booking&) = delete;

    Booking(Booking&& other)
      : provider(other.provider),
        booking_id(other.booking_id)
    {
      other.provider = nullptr;
    }

    Booking& operator = (const Booking&) = delete;

    Booking& operator = (Booking&& other) {
      std::swap(provider, other.provider);
      std::swap(booking_id, other.booking_id);
      return *this;
    }

    // Эта функция не требуется в тестах, но в реальной программе она может быть нужна
    BookingId GetId() const {
      return booking_id;
    }

    ~Booking() {
      if (provider != nullptr) {
        provider->CancelOrComplete(*this);
      }
    }
  };

}
 */