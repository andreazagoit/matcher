"use client";

import dynamic from "next/dynamic";

interface EventMapProps {
  lat: number;
  lon: number;
  label?: string;
  className?: string;
}

const MapInner = dynamic(() => import("./event-map-inner"), { ssr: false });

export function EventMap({ lat, lon, label, className }: EventMapProps) {
  return <MapInner lat={lat} lon={lon} label={label} className={className} />;
}
