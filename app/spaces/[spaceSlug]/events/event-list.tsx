"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@apollo/client/react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { CalendarIcon, PlusIcon, Loader2Icon } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ALL_TAGS } from "@/lib/models/tags/data";
import {
  GET_SPACE_EVENTS,
  CREATE_EVENT,
  RESPOND_TO_EVENT,
  MARK_EVENT_COMPLETED,
} from "@/lib/models/events/gql";
import {
  AttendeeStatus,
  type SpaceEventsQuery,
  type SpaceEventsQueryVariables,
  type CreateEventMutation,
  type CreateEventMutationVariables,
  type RespondToEventMutation,
  type RespondToEventMutationVariables,
  type MarkEventCompletedMutation,
  type MarkEventCompletedMutationVariables,
} from "@/lib/graphql/__generated__/graphql";
import { EventCard, type EventCardEvent } from "@/components/event-card";

interface EventListProps {
  spaceId: string;
  isAdmin: boolean;
}

export function EventList({ spaceId, isAdmin }: EventListProps) {
  const { data, loading, refetch } = useQuery<SpaceEventsQuery, SpaceEventsQueryVariables>(
    GET_SPACE_EVENTS,
    { variables: { spaceId } },
  );

  const [createEvent, { loading: creating }] = useMutation<CreateEventMutation, CreateEventMutationVariables>(CREATE_EVENT);
  const [respondToEvent] = useMutation<RespondToEventMutation, RespondToEventMutationVariables>(RESPOND_TO_EVENT);
  const [markCompleted] = useMutation<MarkEventCompletedMutation, MarkEventCompletedMutationVariables>(MARK_EVENT_COMPLETED);

  const [showCreate, setShowCreate] = useState(false);
  const [newEvent, setNewEvent] = useState({
    title: "",
    description: "",
    location: "",
    startsAt: "",
    endsAt: "",
    tags: [] as string[],
  });

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createEvent({
        variables: {
          input: {
            spaceId,
            title: newEvent.title,
            description: newEvent.description || undefined,
            location: newEvent.location || undefined,
            startsAt: new Date(newEvent.startsAt).toISOString(),
            endsAt: newEvent.endsAt ? new Date(newEvent.endsAt).toISOString() : undefined,
            tags: newEvent.tags.length > 0 ? newEvent.tags : undefined,
          },
        },
      });
      setShowCreate(false);
      setNewEvent({ title: "", description: "", location: "", startsAt: "", endsAt: "", tags: [] });
      refetch();
    } catch (err) {
      console.error("Failed to create event:", err);
    }
  };

  const handleRespond = async (eventId: string, status: AttendeeStatus) => {
    try {
      await respondToEvent({ variables: { eventId, status } });
      refetch();
    } catch (err) {
      console.error("Failed to respond:", err);
    }
  };

  const handleComplete = async (eventId: string) => {
    try {
      await markCompleted({ variables: { eventId } });
      refetch();
    } catch (err) {
      console.error("Failed to complete:", err);
    }
  };

  const events = data?.spaceEvents ?? [];

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2Icon className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Eventi</h3>
        {isAdmin && (
          <Dialog open={showCreate} onOpenChange={setShowCreate}>
            <DialogTrigger asChild>
              <Button size="sm" className="gap-2">
                <PlusIcon className="h-4 w-4" />
                Nuovo evento
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Crea nuovo evento</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleCreate} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="event-title">Titolo</Label>
                  <Input
                    id="event-title"
                    value={newEvent.title}
                    onChange={(e) => setNewEvent((p) => ({ ...p, title: e.target.value }))}
                    required
                    placeholder="Es. Aperitivo al parco"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="event-desc">Descrizione</Label>
                  <Textarea
                    id="event-desc"
                    value={newEvent.description}
                    onChange={(e) => setNewEvent((p) => ({ ...p, description: e.target.value }))}
                    placeholder="Descrivi l'evento..."
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="event-location">Luogo</Label>
                  <Input
                    id="event-location"
                    value={newEvent.location}
                    onChange={(e) => setNewEvent((p) => ({ ...p, location: e.target.value }))}
                    placeholder="Es. Piazza del Duomo, Milano"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Tags</Label>
                  <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                    {ALL_TAGS.map((tag) => {
                      const isSelected = newEvent.tags.includes(tag);
                      return (
                        <Badge
                          key={tag}
                          variant={isSelected ? "default" : "outline"}
                          className="cursor-pointer text-xs py-1 px-2"
                          onClick={() =>
                            setNewEvent((p) => ({
                              ...p,
                              tags: isSelected ? p.tags.filter((t) => t !== tag) : [...p.tags, tag],
                            }))
                          }
                        >
                          {tag}
                        </Badge>
                      );
                    })}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label htmlFor="event-start">Inizio</Label>
                    <Input
                      id="event-start"
                      type="datetime-local"
                      value={newEvent.startsAt}
                      onChange={(e) => setNewEvent((p) => ({ ...p, startsAt: e.target.value }))}
                      required
                      className="dark:[color-scheme:dark]"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="event-end">Fine</Label>
                    <Input
                      id="event-end"
                      type="datetime-local"
                      value={newEvent.endsAt}
                      onChange={(e) => setNewEvent((p) => ({ ...p, endsAt: e.target.value }))}
                      className="dark:[color-scheme:dark]"
                    />
                  </div>
                </div>
                <Button type="submit" disabled={creating} className="w-full">
                  {creating ? (
                    <Loader2Icon className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <PlusIcon className="h-4 w-4 mr-2" />
                  )}
                  Crea evento
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {events.length === 0 ? (
        <div className="text-center py-16 rounded-2xl border-2 border-dashed border-muted-foreground/20">
          <CalendarIcon className="h-10 w-10 mx-auto mb-3 text-muted-foreground/40" />
          <p className="font-medium text-muted-foreground">Nessun evento ancora</p>
          {isAdmin && (
            <p className="text-sm text-muted-foreground/70 mt-1">
              Crea il primo evento per questa community!
            </p>
          )}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {events.map((event) => (
            <EventCard
              key={event.id}
              event={event as EventCardEvent}
              onRespond={handleRespond}
              onComplete={isAdmin ? handleComplete : undefined}
            />
          ))}
        </div>
      )}
    </div>
  );
}
