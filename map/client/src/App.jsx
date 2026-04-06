import "./App.css";
import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  MapContainer,
  Marker,
  Polyline,
  TileLayer,
  Popup,
  ZoomControl,
  useMap,
  useMapEvents,
} from "react-leaflet";
import {
  FaArrowRightArrowLeft,
  FaLocationCrosshairs,
  FaMapLocationDot,
  FaRoute,
} from "react-icons/fa6";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const BACKEND_URL = "http://localhost:8000/start";
const GEOCODE_URL = "https://nominatim.openstreetmap.org/search";
const REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse";
const ROUTE_API_URL = "https://router.project-osrm.org/route/v1/driving";
const DEFAULT_CENTER = [26.9124, 75.7873];

const MAP_MODES = {
  street: {
    name: "Street",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },
  satellite: {
    name: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "&copy; Esri",
  },
};

const markerIcon = (color) =>
  L.divIcon({
    className: "custom-marker",
    html: `<div style=\"width:22px;height:22px;border-radius:999px;background:${color};border:3px solid white;box-shadow:0 6px 15px rgba(0,0,0,0.22);\"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });

function FitRouteToBounds({ routePath }) {
  const map = useMap();

  useEffect(() => {
    if (!routePath || routePath.length < 2) {
      return;
    }

    map.fitBounds(routePath, {
      padding: [70, 70],
      maxZoom: 15,
    });
  }, [map, routePath]);

  return null;
}

function DestinationPinPicker({ enabled, onPick }) {
  useMapEvents({
    click: (event) => {
      if (!enabled) {
        return;
      }

      onPick([event.latlng.lat, event.latlng.lng]);
    },
  });

  return null;
}

const haversineDistanceKm = (start, end) => {
  if (!start || !end) {
    return 0;
  }

  const toRad = (degrees) => (degrees * Math.PI) / 180;
  const [lat1, lon1] = start;
  const [lat2, lon2] = end;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const lat1Rad = toRad(lat1);
  const lat2Rad = toRad(lat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1Rad) *
    Math.cos(lat2Rad) *
    Math.sin(dLon / 2) *
    Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return 6371 * c;
};

const formatEta = (minutes) => {
  const rounded = Math.max(1, Math.round(minutes));
  if (rounded < 60) {
    return `${rounded} min`;
  }

  const hours = Math.floor(rounded / 60);
  const mins = rounded % 60;
  return mins > 0 ? `${hours} h ${mins} min` : `${hours} h`;
};

function App() {
  const [mapMode, setMapMode] = useState("street");
  const [startAddress, setStartAddress] = useState("");
  const [endAddress, setEndAddress] = useState("");
  const [startSuggestions, setStartSuggestions] = useState([]);
  const [endSuggestions, setEndSuggestions] = useState([]);
  const [activeInput, setActiveInput] = useState(null);
  const [isStartCoordsSynced, setIsStartCoordsSynced] = useState(false);
  const [isEndCoordsSynced, setIsEndCoordsSynced] = useState(false);
  const [startCoords, setStartCoords] = useState(null);
  const [endCoords, setEndCoords] = useState(null);
  const [routePath, setRoutePath] = useState([]);
  const [routeSummary, setRouteSummary] = useState(null);
  const [isPinMode, setIsPinMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [locating, setLocating] = useState(false);
  const [error, setError] = useState("");
  const [backendError, setBackendError] = useState("");
  const [backendResult, setBackendResult] = useState(null);

  const currentMap = MAP_MODES[mapMode];

  const mapCenter = useMemo(() => {
    if (startCoords && endCoords) {
      const lat = (startCoords[0] + endCoords[0]) / 2;
      const lon = (startCoords[1] + endCoords[1]) / 2;
      return [lat, lon];
    }

    return startCoords || endCoords || DEFAULT_CENTER;
  }, [startCoords, endCoords]);

  const reverseGeocode = async (lat, lon) => {
    const { data } = await axios.get(REVERSE_GEOCODE_URL, {
      params: { lat, lon, format: "json" },
      headers: {
        Accept: "application/json",
      },
    });

    return data?.display_name || `Lat ${lat}, Lon ${lon}`;
  };

  const geocodeAddress = async (address) => {
    const { data } = await axios.get(GEOCODE_URL, {
      params: {
        q: address,
        format: "json",
        limit: 1,
      },
      headers: {
        Accept: "application/json",
      },
    });

    if (!data?.length) {
      throw new Error(`Address not found: ${address}`);
    }

    return [Number(data[0].lat), Number(data[0].lon)];
  };

  const fetchPlaceSuggestions = async (query, signal) => {
    const { data } = await axios.get(GEOCODE_URL, {
      params: {
        q: query,
        format: "json",
        addressdetails: 1,
        limit: 5,
      },
      signal,
      headers: {
        Accept: "application/json",
      },
    });

    return (data || []).map((item) => ({
      id: item.place_id,
      label: item.display_name,
      lat: Number(item.lat),
      lon: Number(item.lon),
    }));
  };

  const detectCurrentLocation = () => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported in this browser.");
      return;
    }

    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        console.log("Current location coords:", { lat, lon });
        setStartCoords([lat, lon]);

        try {
          const label = await reverseGeocode(lat, lon);
          setStartAddress(label);
          setIsStartCoordsSynced(true);
        } catch {
          setStartAddress(`Lat ${lat.toFixed(5)}, Lon ${lon.toFixed(5)}`);
          setIsStartCoordsSynced(true);
        } finally {
          setLocating(false);
        }
      },
      () => {
        setLocating(false);
        setError("Could not access current location.");
      },
      { enableHighAccuracy: true, timeout: 10000 },
    );
  };

  useEffect(() => {
    const query = startAddress.trim();
    if (!query || query.length < 3) {
      setStartSuggestions([]);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const suggestions = await fetchPlaceSuggestions(
          query,
          controller.signal,
        );
        setStartSuggestions(suggestions);
      } catch (requestError) {
        if (requestError.name !== "CanceledError") {
          setStartSuggestions([]);
        }
      }
    }, 350);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [startAddress]);

  useEffect(() => {
    const query = endAddress.trim();
    if (!query || query.length < 3) {
      setEndSuggestions([]);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const suggestions = await fetchPlaceSuggestions(
          query,
          controller.signal,
        );
        setEndSuggestions(suggestions);
      } catch (requestError) {
        if (requestError.name !== "CanceledError") {
          setEndSuggestions([]);
        }
      }
    }, 350);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [endAddress]);

  useEffect(() => {
    if (
      !startCoords ||
      !endCoords ||
      !isStartCoordsSynced ||
      !isEndCoordsSynced
    ) {
      setRoutePath([]);
      setRouteSummary(null);
      return;
    }

    const controller = new AbortController();

    const fetchExpectedRoute = async () => {
      try {
        const [startLat, startLon] = startCoords;
        const [endLat, endLon] = endCoords;
        const routeUrl = `${ROUTE_API_URL}/${startLon},${startLat};${endLon},${endLat}`;
        const { data } = await axios.get(routeUrl, {
          params: {
            overview: "full",
            geometries: "geojson",
          },
          signal: controller.signal,
        });

        const geometry = data?.routes?.[0]?.geometry?.coordinates;
        const distanceMeters = data?.routes?.[0]?.distance;
        const durationSeconds = data?.routes?.[0]?.duration;

        if (distanceMeters && durationSeconds) {
          setRouteSummary({
            distanceKm: distanceMeters / 1000,
            durationMinutes: durationSeconds / 60,
          });
        }

        if (Array.isArray(geometry) && geometry.length > 0) {
          setRoutePath(geometry.map(([lon, lat]) => [lat, lon]));
          return;
        }

        if (!distanceMeters || !durationSeconds) {
          const distanceKm = haversineDistanceKm(startCoords, endCoords);
          setRouteSummary({
            distanceKm,
            durationMinutes: (distanceKm / 45) * 60,
          });
        }
        setRoutePath([startCoords, endCoords]);
      } catch (requestError) {
        if (requestError.name !== "CanceledError") {
          const distanceKm = haversineDistanceKm(startCoords, endCoords);
          setRoutePath([startCoords, endCoords]);
          setRouteSummary({
            distanceKm,
            durationMinutes: (distanceKm / 45) * 60,
          });
        }
      }
    };

    fetchExpectedRoute();

    return () => {
      controller.abort();
    };
  }, [startCoords, endCoords, isStartCoordsSynced, isEndCoordsSynced]);

  const switchNodes = () => {
    const nextStartSynced = isEndCoordsSynced;
    const nextEndSynced = isStartCoordsSynced;

    setStartAddress(endAddress || "");
    setEndAddress(startAddress || "");
    setStartCoords(endCoords);
    setEndCoords(startCoords);
    setIsStartCoordsSynced(nextStartSynced);
    setIsEndCoordsSynced(nextEndSynced);
    setIsPinMode(false);
    setStartSuggestions([]);
    setEndSuggestions([]);
  };

  const handleDestinationPin = async (coords) => {
    const [lat, lon] = coords;

    setEndCoords(coords);
    setIsEndCoordsSynced(true);
    setIsPinMode(false);
    setEndSuggestions([]);
    setActiveInput(null);
    setError("");

    try {
      const label = await reverseGeocode(lat, lon);
      setEndAddress(label);
    } catch {
      setEndAddress(`Pinned destination (${lat.toFixed(5)}, ${lon.toFixed(5)})`);
    }
  };

  const selectSuggestion = (type, suggestion) => {
    if (type === "start") {
      setStartAddress(suggestion.label);
      setStartCoords([suggestion.lat, suggestion.lon]);
      setIsStartCoordsSynced(true);
      setStartSuggestions([]);
    } else {
      setEndAddress(suggestion.label);
      setEndCoords([suggestion.lat, suggestion.lon]);
      setIsEndCoordsSynced(true);
      setEndSuggestions([]);
    }

    setActiveInput(null);
  };

  const handleDirectionRequest = async (event) => {
    event.preventDefault();
    setError("");
    setBackendError("");
    setBackendResult(null);
    setLoading(true);

    try {
      const hasStartAddress = Boolean(startAddress.trim());
      if (!hasStartAddress && !startCoords) {
        throw new Error("Please enter a start address or click locate.");
      }

      const finalStart = isStartCoordsSynced
        ? startCoords
        : await geocodeAddress(startAddress);

      if (!endAddress.trim()) {
        throw new Error("Please enter destination address.");
      }

      const finalEnd = isEndCoordsSynced
        ? endCoords
        : await geocodeAddress(endAddress);
      console.log("Direction request coords:", {
        start_lat: finalStart[0],
        start_lon: finalStart[1],
        end_lat: finalEnd[0],
        end_lon: finalEnd[1],
      });
      setStartCoords(finalStart);
      setEndCoords(finalEnd);
      setIsStartCoordsSynced(true);
      setIsEndCoordsSynced(true);

      try {
        const response = await axios.get(BACKEND_URL, {
          params: {
            start_lat: finalStart[0],
            start_lon: finalStart[1],
            end_lat: finalEnd[0],
            end_lon: finalEnd[1],
          },
        });

        const responseData = response.data;
        
        // SNAP TO ROAD FIX: If Python proxy blocks the OSRM route and passes math vectors, dynamically overwrite it in UI using the user's browser!
        // We MUST pass the AI's intermediate points into OSRM so it snaps the AI's chosen alternative shape to real roads, instead of defaulting to Google-like shortest path!
        if (responseData.best_route_geometry && responseData.best_route_geometry.length <= 20) {
          try {
            // Join all AI intermediate fallback nodes. Leaflet is [lat, lon], OSRM expects [lon, lat]
            const aiPointsStr = responseData.best_route_geometry.map((c) => `${c[1]},${c[0]}`).join(";");
            const osrmURL = `${ROUTE_API_URL}/${aiPointsStr}?overview=full&geometries=geojson`;
            
            const osrmResponse = await axios.get(osrmURL);
            if (osrmResponse.data.routes && osrmResponse.data.routes.length > 0) {
              const coords = osrmResponse.data.routes[0].geometry.coordinates;
              responseData.best_route_geometry = coords.map((c) => [c[1], c[0]]);
            }
          } catch (e) {
             console.error("Frontend OSRM failed too", e);
          }
        }

        setBackendResult(responseData);
      } catch {
        setBackendError(
          "Backend is unreachable right now. Showing map route, distance, and ETA.",
        );
      }
    } catch (requestError) {
      setError(requestError.message || "Request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative h-screen w-full overflow-hidden bg-slate-200">
      <MapContainer
        center={mapCenter}
        zoom={13}
        zoomControl={false}
        attributionControl={false}
        className="h-full w-full"
      >
        <TileLayer attribution={currentMap.attribution} url={currentMap.url} />
        <ZoomControl position="bottomright" />
        <FitRouteToBounds routePath={routePath} />
        <DestinationPinPicker enabled={isPinMode} onPick={handleDestinationPin} />
        {backendResult?.best_route_geometry ? (
          <>
            <Polyline
              positions={backendResult.best_route_geometry}
              pathOptions={{ color: "#ffffff", weight: 12, opacity: 0.8 }}
            />
            <Polyline
              positions={backendResult.best_route_geometry}
              pathOptions={{ color: "#10b981", weight: 7, opacity: 0.95 }}
            />
          </>
        ) : routePath.length > 1 ? (
          <>
            <Polyline
              positions={routePath}
              pathOptions={{ color: "#ffffff", weight: 10, opacity: 0.65 }}
            />
            <Polyline
              positions={routePath}
              pathOptions={{ color: "#2563eb", weight: 6, opacity: 0.95 }}
            />
          </>
        ) : null}
        {startCoords ? (
          <Marker position={startCoords} icon={markerIcon("#22c55e")}>
            <Popup>Start: {startAddress || "Start Location"}</Popup>
          </Marker>
        ) : null}
        {endCoords ? (
          <Marker position={endCoords} icon={markerIcon("#ef4444")}>
            <Popup>End: {endAddress || "Destination"}</Popup>
          </Marker>
        ) : null}
      </MapContainer>

      <button
        type="button"
        onClick={() =>
          setMapMode((prev) => (prev === "street" ? "satellite" : "street"))
        }
        className="absolute right-5 top-5 z-1000 rounded-full bg-white/95 px-4 py-2 text-sm font-semibold text-slate-700 shadow-xl ring-1 ring-slate-200"
      >
        Switch Map: {currentMap.name}
      </button>

      <main className="pointer-events-none absolute inset-0 z-950 flex items-start justify-start p-4 md:p-6">
        <section className="pointer-events-auto w-full max-w-sm rounded-3xl bg-white/92 p-4 shadow-2xl backdrop-blur-md md:p-5">
          <h1 className="mb-4 text-left text-3xl font-black tracking-tight text-slate-800">
            Plan Your Trip
          </h1>

          <form onSubmit={handleDirectionRequest} className="space-y-3">
            <div className="rounded-2xl border border-slate-200 bg-white px-3 py-3">
              <div className="mb-2 flex items-center gap-3 text-base font-semibold text-slate-800">
                <span className="h-4 w-4 rounded-full bg-emerald-500" />
                Start
              </div>
              <div className="relative flex items-center gap-3">
                <FaMapLocationDot className="text-lg text-blue-500" />
                <input
                  className="w-full bg-transparent text-base text-slate-500 outline-none"
                  value={startAddress}
                  onChange={(event) => {
                    setStartAddress(event.target.value);
                    setIsStartCoordsSynced(false);
                    setActiveInput("start");
                  }}
                  onFocus={() => setActiveInput("start")}
                  onBlur={() => setTimeout(() => setActiveInput(null), 120)}
                  placeholder="Enter start location"
                />
                <button
                  type="button"
                  className="rounded-full border border-slate-200 p-2 text-slate-500 transition hover:bg-slate-50"
                  onClick={detectCurrentLocation}
                  title="Use my current location"
                >
                  <FaLocationCrosshairs />
                </button>

                {activeInput === "start" && startSuggestions.length > 0 ? (
                  <ul className="absolute left-8 right-12 top-10 z-20 max-h-48 overflow-auto rounded-xl border border-slate-200 bg-white py-2 shadow-lg">
                    {startSuggestions.map((suggestion) => (
                      <li
                        key={suggestion.id}
                        className="cursor-pointer px-3 py-2 text-sm text-slate-700 hover:bg-slate-100"
                        onMouseDown={() =>
                          selectSuggestion("start", suggestion)
                        }
                      >
                        {suggestion.label}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </div>

            <div className="flex justify-center">
              <button
                type="button"
                onClick={switchNodes}
                className="rounded-full border border-slate-200 bg-white p-2.5 text-slate-500 shadow"
                title="Switch start and end"
              >
                <FaArrowRightArrowLeft />
              </button>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white px-3 py-3">
              <div className="mb-2 flex items-center gap-3 text-base font-semibold text-slate-800">
                <span className="h-4 w-4 rounded-full bg-red-500" />
                End
              </div>
              <div className="relative flex items-center gap-3">
                <FaMapLocationDot className="text-lg text-red-500" />
                <input
                  className="w-full bg-transparent text-base text-slate-500 outline-none"
                  value={endAddress}
                  onChange={(event) => {
                    setEndAddress(event.target.value);
                    setIsEndCoordsSynced(false);
                    setActiveInput("end");
                  }}
                  onFocus={() => setActiveInput("end")}
                  onBlur={() => setTimeout(() => setActiveInput(null), 120)}
                  placeholder="Enter destination"
                />

                {activeInput === "end" && endSuggestions.length > 0 ? (
                  <ul className="absolute left-8 right-0 top-10 z-20 max-h-48 overflow-auto rounded-xl border border-slate-200 bg-white py-2 shadow-lg">
                    {endSuggestions.map((suggestion) => (
                      <li
                        key={suggestion.id}
                        className="cursor-pointer px-3 py-2 text-sm text-slate-700 hover:bg-slate-100"
                        onMouseDown={() => selectSuggestion("end", suggestion)}
                      >
                        {suggestion.label}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => setIsPinMode((prev) => !prev)}
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
              >
                {isPinMode ? "Click on map to pin destination..." : "Use Pin for Destination"}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading || locating}
              className="flex w-full items-center justify-center gap-2 rounded-full bg-blue-500 py-3 text-lg font-bold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <FaRoute />
              {loading
                ? "Loading..."
                : locating
                  ? "Locating..."
                  : "Get Directions"}
            </button>

            {error ? (
              <p className="text-sm font-semibold text-red-500">{error}</p>
            ) : null}
            {backendError ? (
              <p className="text-sm font-semibold text-amber-600">{backendError}</p>
            ) : null}
            {routeSummary ? (
              <div className="rounded-xl border border-blue-100 bg-blue-50 px-3 py-2 text-sm text-slate-700">
                <p className="font-semibold text-blue-700">Trip Summary</p>
                <p>
                  Distance: {routeSummary.distanceKm.toFixed(1)} km
                </p>
                <p>
                  Estimated Car Time: {formatEta(routeSummary.durationMinutes)}
                </p>
              </div>
            ) : null}
            {backendResult ? (
              <div className="mt-4 flex flex-col gap-3 rounded-2xl bg-slate-800 p-5 text-white shadow-xl">
                <div className="flex items-center justify-between border-b border-slate-700 pb-3">
                  <h3 className="text-lg font-bold text-emerald-400">SafHer Route Evaluated</h3>
                  <span className="rounded-full bg-slate-700 px-3 py-1 font-mono text-xs font-semibold">
                    Driver: {backendResult.selected_driver_id}
                  </span>
                </div>
                
                <div className="grid grid-cols-3 gap-2 text-center mt-2">
                  <div className="rounded-xl bg-slate-900 p-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-400">Driver Risk</div>
                    <div className="font-bold text-blue-300">{backendResult.driver_risk_isolated?.toFixed(2)}</div>
                  </div>
                  <div className="rounded-xl bg-slate-900 p-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-400">Route Risk</div>
                    <div className="font-bold text-emerald-300">{backendResult.route_risk_isolated?.toFixed(2)}</div>
                  </div>
                  <div className="rounded-xl bg-slate-900 p-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-400">Context Risk</div>
                    <div className="font-bold text-amber-300">{backendResult.context_risk_isolated?.toFixed(2)}</div>
                  </div>
                </div>

                <div className="mt-2 flex items-center justify-between px-1">
                  <span className="text-sm font-semibold text-slate-300">Total Safety Score</span>
                  <span className="text-2xl font-black text-emerald-400">{backendResult.combinatorial_risk_score}</span>
                </div>

                {backendResult.contextual_alerts && backendResult.contextual_alerts.length > 0 && (
                  <div className="mt-2 rounded-xl bg-red-900/40 p-3 pt-2 outline outline-1 outline-red-500/50">
                    <h4 className="mb-1 text-xs font-bold text-red-400">⚠️ Contextual Alerts</h4>
                    <ul className="list-inside list-disc text-xs text-red-200">
                      {backendResult.contextual_alerts.map((alert, i) => (
                        <li key={i} className="mb-1">{alert}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : null}
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;