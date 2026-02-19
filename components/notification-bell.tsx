"use client";

import { useQuery, useMutation } from "@apollo/client/react";
import { Bell } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  GET_NOTIFICATIONS,
  GET_UNREAD_COUNT,
  MARK_NOTIFICATION_READ,
  MARK_ALL_NOTIFICATIONS_READ,
} from "@/lib/models/notifications/gql";
import type {
  GetNotificationsQuery,
  GetUnreadNotificationsCountQuery,
} from "@/lib/graphql/__generated__/graphql";
import { useSession } from "@/lib/auth-client";
import { cn } from "@/lib/utils";

export function NotificationBell() {
  const { data: session } = useSession();
  const router = useRouter();

  const { data: countData, refetch: refetchCount } =
    useQuery<GetUnreadNotificationsCountQuery>(GET_UNREAD_COUNT, {
      skip: !session?.user,
      pollInterval: 30000,
    });

  const { data, refetch: refetchList } = useQuery<GetNotificationsQuery>(
    GET_NOTIFICATIONS,
    { skip: !session?.user, variables: { limit: 8 } },
  );

  const [markRead] = useMutation(MARK_NOTIFICATION_READ, {
    onCompleted: () => { refetchCount(); refetchList(); },
  });

  const [markAllRead] = useMutation(MARK_ALL_NOTIFICATIONS_READ, {
    onCompleted: () => { refetchCount(); refetchList(); },
  });

  if (!session?.user) return null;

  const unread = countData?.unreadNotificationsCount ?? 0;
  const notifications = data?.notifications ?? [];

  const handleClick = async (id: string, href?: string | null) => {
    await markRead({ variables: { id } });
    if (href) router.push(href);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="relative h-9 w-9 p-0">
          <Bell className="h-4 w-4" />
          {unread > 0 && (
            <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[10px] font-bold text-destructive-foreground">
              {unread > 9 ? "9+" : unread}
            </span>
          )}
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel className="flex items-center justify-between">
          <span>Notifiche</span>
          {unread > 0 && (
            <button
              onClick={() => markAllRead()}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Segna tutte come lette
            </button>
          )}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />

        {notifications.length === 0 ? (
          <div className="py-6 text-center text-sm text-muted-foreground">
            Nessuna notifica
          </div>
        ) : (
          notifications.map((n) => (
            <DropdownMenuItem
              key={n.id}
              className={cn(
                "flex items-start gap-3 cursor-pointer py-3",
                !n.read && "bg-muted/50",
              )}
              onClick={() => handleClick(n.id, n.href)}
            >
              {n.image ? (
                <img
                  src={n.image}
                  alt=""
                  className="h-8 w-8 rounded-full object-cover shrink-0 mt-0.5"
                />
              ) : (
                <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Bell className="h-3.5 w-3.5 text-primary" />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className={cn("text-sm leading-snug", !n.read && "font-medium")}>
                  {n.text}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {new Date(n.createdAt as string).toLocaleDateString("it-IT", {
                    day: "numeric",
                    month: "short",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>
              {!n.read && (
                <div className="h-2 w-2 rounded-full bg-primary shrink-0 mt-1.5" />
              )}
            </DropdownMenuItem>
          ))
        )}

        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link href="/notifications" className="justify-center text-sm text-muted-foreground">
            Vedi tutte
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
