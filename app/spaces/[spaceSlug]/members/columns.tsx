"use client"

import { ColumnDef } from "@tanstack/react-table"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    EllipsisVerticalIcon,
    ShieldIcon,
    ArrowUpDown,
    UserIcon,
} from "lucide-react"
import type { Member } from "@/lib/graphql/__generated__/graphql"

export type { Member }

export const columns: ColumnDef<Member>[] = [
    {
        id: "select",
        header: ({ table }) => (
            <Checkbox
                checked={
                    table.getIsAllPageRowsSelected() ||
                    (table.getIsSomePageRowsSelected() && "indeterminate")
                }
                onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
                aria-label="Select all"
            />
        ),
        cell: ({ row }) => (
            <Checkbox
                checked={row.getIsSelected()}
                onCheckedChange={(value) => row.toggleSelected(!!value)}
                aria-label="Select row"
                onClick={(e) => e.stopPropagation()}
            />
        ),
        enableSorting: false,
        enableHiding: false,
    },
    {
        id: "user",
        accessorFn: (row) => `${row.user.givenName} ${row.user.familyName} ${row.user.email}`,
        header: ({ column }) => {
            return (
                <Button
                    variant="ghost"
                    onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                    className="-ml-4"
                >
                    User
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
            )
        },
        cell: ({ row }) => {
            const member = row.original
            return (
                <div className="flex items-center gap-3">
                    <Avatar className="h-9 w-9">
                        <AvatarFallback className="bg-primary/10 text-primary text-sm">
                            {member.user.givenName[0]}{member.user.familyName[0]}
                        </AvatarFallback>
                    </Avatar>
                    <div>
                        <p className="font-medium text-sm">{member.user.givenName} {member.user.familyName}</p>
                        <p className="text-xs text-muted-foreground">{member.user.email}</p>
                    </div>
                </div>
            )
        },
    },
    {
        accessorKey: "role",
        header: "Role",
        cell: ({ row }) => {
            const role = row.getValue("role") as string
            return (
                <>
                    {role === "admin" && (
                        <Badge variant="default" className="gap-1">
                            <ShieldIcon className="h-3 w-3" />
                            Admin
                        </Badge>
                    )}
                    {role === "member" && (
                        <Badge variant="outline" className="gap-1">
                            <UserIcon className="h-3 w-3" />
                            Member
                        </Badge>
                    )}
                </>
            )
        },
        filterFn: (row, id, value) => {
            return value.includes(row.getValue(id))
        },
    },
    {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => {
            const status = row.getValue("status") as string
            return (
                <div className="flex items-center gap-2">
                    <div
                        className={`h-2 w-2 rounded-full ${status === "active"
                            ? "bg-green-500"
                            : status === "pending"
                                ? "bg-yellow-500"
                                : "bg-red-500"
                            }`}
                    />
                    <span className="capitalize text-sm">{status}</span>
                </div>
            )
        },
    },
    {
        id: "tier",
        accessorFn: (row) => row.tier?.name,
        header: "Tier",
        cell: ({ row }) => {
            const tierName = row.original.tier?.name
            return tierName ? (
                <Badge variant="secondary" className="font-normal border-primary/20 bg-primary/5 text-primary">
                    {tierName}
                </Badge>
            ) : (
                <span className="text-muted-foreground text-xs pl-2">-</span>
            )
        }
    },
    {
        accessorKey: "joinedAt",
        header: ({ column }) => {
            return (
                <Button
                    variant="ghost"
                    onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                    className="-ml-4"
                >
                    Joined
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
            )
        },
        cell: ({ row }) => {
            const date = new Date(row.getValue("joinedAt"))
            return (
                <span className="text-muted-foreground text-sm">
                    {date.toLocaleDateString(undefined, {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                    })}
                </span>
            )
        },
    },
    {
        id: "actions",
        cell: ({ row }) => {
            const member = row.original

            return (
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                            <EllipsisVerticalIcon className="h-4 w-4" />
                            <span className="sr-only">Open menu</span>
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Actions</DropdownMenuLabel>
                        <DropdownMenuItem
                            onClick={(e) => {
                                e.stopPropagation()
                                navigator.clipboard.writeText(member.user.email)
                            }}
                        >
                            Copy email
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem disabled className="text-muted-foreground">
                            Use details modal to change role
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            )
        },
    },
]
