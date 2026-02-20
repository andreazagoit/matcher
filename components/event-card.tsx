"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CalendarIcon, MapPinIcon, UsersIcon, CheckCircleIcon } from "lucide-react";
import { AttendeeStatus } from "@/lib/graphql/__generated__/graphql";

const TAG_GRADIENTS: Record<string, string> = {
  trekking: "from-emerald-500 to-teal-700",
  mountains: "from-slate-500 to-slate-800",
  cycling: "from-lime-500 to-green-700",
  running: "from-orange-400 to-red-600",
  yoga: "from-violet-400 to-purple-700",
  gym: "from-blue-500 to-indigo-700",
  cooking: "from-amber-400 to-orange-600",
  wine: "from-rose-500 to-red-800",
  restaurants: "from-pink-400 to-rose-600",
  music: "from-purple-500 to-violet-800",
  art: "from-fuchsia-400 to-pink-700",
  coding: "from-cyan-500 to-blue-700",
  gaming: "from-indigo-400 to-purple-600",
  parties: "from-yellow-400 to-orange-500",
  cinema: "from-neutral-500 to-neutral-800",
  reading: "from-stone-400 to-stone-700",
  travel: "from-sky-400 to-blue-600",
};

export function getEventGradient(tags: string[]) {
  for (const tag of tags ?? []) {
    if (TAG_GRADIENTS[tag]) return TAG_GRADIENTS[tag];
  }
  return "from-primary/60 to-primary";
}

export interface EventCardEvent {
  id: string;
  title: string;
  description?: string | null;
  location?: string | null;
  startsAt: string;
  endsAt?: string | null;
  attendeeCount: number;
  maxAttendees?: number | null;
  tags: string[];
  status?: string | null;
}

interface EventCardProps {
  event: EventCardEvent;
  onRespond?: (id: string, status: AttendeeStatus) => void;
  onComplete?: (id: string) => void;
}

export function EventCard({ event, onRespond, onComplete }: EventCardProps) {
  const startDate = new Date(event.startsAt);
  const isPast = startDate < new Date();
  const isPublished = event.status === "published";
  const isCompleted = event.status === "completed";
  const gradient = getEventGradient(event.tags ?? []);

  return (
    <div className={`flex flex-col gap-3 group ${isCompleted ? "opacity-60" : ""}`}>
      <Link href={`/events/${event.id}`} className="block">
        <div className="aspect-square w-full relative bg-muted rounded-xl overflow-hidden ring-1 ring-border/50 group-hover:ring-primary/20 transition-all">
          <div className={`size-full bg-gradient-to-br ${gradient} flex items-center justify-center`}>
            <span className="text-6xl font-black text-white/20 select-none group-hover:scale-110 transition-transform">
              {event.title.charAt(0).toUpperCase()}
            </span>
          </div>

          <div className="absolute top-2 right-2">
            <Badge className="bg-white/90 text-black hover:bg-white shadow-sm backdrop-blur-sm h-6 gap-1 font-medium">
              <CalendarIcon className="h-3 w-3" />
              {startDate.toLocaleDateString("it-IT", { day: "numeric", month: "short" })}
            </Badge>
          </div>

          <div className="absolute bottom-2 left-2">
            <Badge className="bg-black/60 text-white hover:bg-black/70 backdrop-blur-sm h-6 font-mono">
              {startDate.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" })}
            </Badge>
          </div>
        </div>
      </Link>

      <div className="space-y-1.5 px-1">
        <h3 className="font-semibold tracking-tight text-lg leading-tight group-hover:text-primary transition-colors line-clamp-1">
          {event.title}
        </h3>

        <p className="text-sm text-muted-foreground line-clamp-1">
          {event.description || (event.location ?? "Nessuna descrizione")}
        </p>

        <div className="flex items-center gap-3 pt-1 text-xs text-muted-foreground">
          {event.location && (
            <div className="flex items-center gap-1.5 min-w-0">
              <MapPinIcon className="size-3.5 shrink-0" />
              <span className="truncate">{event.location}</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 shrink-0 ml-auto">
            <UsersIcon className="size-3.5" />
            <span>{event.attendeeCount}{event.maxAttendees ? `/${event.maxAttendees}` : ""}</span>
          </div>
        </div>

        {(onRespond || onComplete) && (
          <div className="flex gap-2 pt-1">
            {onRespond && isPublished && !isPast && (
              <>
                <Button size="sm" className="flex-1" onClick={() => onRespond(event.id, AttendeeStatus.Going)}>
                  Partecipo
                </Button>
                <Button size="sm" variant="outline" onClick={() => onRespond(event.id, AttendeeStatus.Interested)}>
                  Interessato
                </Button>
              </>
            )}
            {onComplete && isPublished && isPast && !isCompleted && (
              <Button size="sm" variant="secondary" className="gap-1 w-full" onClick={() => onComplete(event.id)}>
                <CheckCircleIcon className="h-3.5 w-3.5" />
                Segna completato
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
