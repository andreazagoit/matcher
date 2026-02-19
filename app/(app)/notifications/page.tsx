"use client";

import { useQuery, useMutation } from "@apollo/client/react";
import { Bell, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { Page } from "@/components/page";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  GET_NOTIFICATIONS,
  GET_UNREAD_COUNT,
  MARK_NOTIFICATION_READ,
  MARK_ALL_NOTIFICATIONS_READ,
  DELETE_NOTIFICATION,
} from "@/lib/models/notifications/gql";
import type { GetNotificationsQuery } from "@/lib/graphql/__generated__/graphql";

export default function NotificationsPage() {
  const router = useRouter();

  const { data, refetch } = useQuery<GetNotificationsQuery>(GET_NOTIFICATIONS, {
    variables: { limit: 50 },
  });

  const { refetch: refetchCount } = useQuery(GET_UNREAD_COUNT);

  const [markRead] = useMutation(MARK_NOTIFICATION_READ, {
    onCompleted: () => { refetch(); refetchCount(); },
  });

  const [markAllRead] = useMutation(MARK_ALL_NOTIFICATIONS_READ, {
    onCompleted: () => { refetch(); refetchCount(); },
  });

  const [deleteNotification] = useMutation(DELETE_NOTIFICATION, {
    onCompleted: () => { refetch(); refetchCount(); },
  });

  const notifications = data?.notifications ?? [];
  const unread = notifications.filter((n) => !n.read).length;

  const handleClick = async (id: string, href?: string | null) => {
    await markRead({ variables: { id } });
    if (href) router.push(href);
  };

  return (
    <Page
      breadcrumbs={[{ label: "Notifiche" }]}
      header={
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h1 className="text-4xl font-extrabold tracking-tight">Notifiche</h1>
            <p className="text-lg text-muted-foreground font-medium">
              {unread > 0 ? `${unread} non lette` : "Tutto aggiornato"}
            </p>
          </div>
          {unread > 0 && (
            <Button variant="outline" size="sm" onClick={() => markAllRead()}>
              Segna tutte come lette
            </Button>
          )}
        </div>
      }
    >
      {notifications.length === 0 ? (
        <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
          <Bell className="h-12 w-12 mx-auto mb-4 text-muted-foreground/40" />
          <h3 className="text-xl font-semibold">Nessuna notifica</h3>
          <p className="text-muted-foreground mt-2">
            Quando riceverai notifiche, appariranno qui.
          </p>
        </div>
      ) : (
        <div className="space-y-1">
          {notifications.map((n) => (
            <div
              key={n.id}
              className={cn(
                "flex items-start gap-4 p-4 rounded-xl transition-colors group",
                !n.read ? "bg-muted/50" : "hover:bg-muted/30",
                n.href && "cursor-pointer",
              )}
              onClick={() => n.href && handleClick(n.id, n.href)}
            >
              {n.image ? (
                <img
                  src={n.image}
                  alt=""
                  className="h-10 w-10 rounded-full object-cover shrink-0 mt-0.5"
                />
              ) : (
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Bell className="h-4 w-4 text-primary" />
                </div>
              )}

              <div className="flex-1 min-w-0">
                <p className={cn("text-sm leading-snug", !n.read && "font-medium")}>
                  {n.text}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {new Date(n.createdAt as string).toLocaleDateString("it-IT", {
                    day: "numeric",
                    month: "long",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>

              <div className="flex items-center gap-2 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                {!n.read && (
                  <button
                    onClick={(e) => { e.stopPropagation(); markRead({ variables: { id: n.id } }); }}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Segna come letta
                  </button>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); deleteNotification({ variables: { id: n.id } }); }}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>

              {!n.read && (
                <div className="h-2 w-2 rounded-full bg-primary shrink-0 mt-2" />
              )}
            </div>
          ))}
        </div>
      )}
    </Page>
  );
}
