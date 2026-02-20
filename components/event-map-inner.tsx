"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix default marker icons broken by webpack
delete (L.Icon.Default.prototype as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

interface Props {
  lat: number;
  lon: number;
  label?: string;
  className?: string;
}

export default function EventMapInner({ lat, lon, label, className }: Props) {
  return (
    <div className={className ?? "h-48 w-full rounded-xl overflow-hidden"}>
      <MapContainer
        center={[lat, lon]}
        zoom={14}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {label && (
          <Marker position={[lat, lon]}>
            <Popup>{label}</Popup>
          </Marker>
        )}
        {!label && <Marker position={[lat, lon]} />}
      </MapContainer>
    </div>
  );
}
