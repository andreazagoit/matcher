"use client"

import { useState } from "react"
import { DataTable } from "@/components/ui/data-table"
import { columns, type Member } from "./columns"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import {
    ShieldIcon,
    UserIcon,
    MailIcon,
    CalendarIcon,
    CircleIcon,
    Loader2Icon,
    TrashIcon,
    UsersIcon,
} from "lucide-react"
import { useMutation } from "@apollo/client/react"
import { UPDATE_MEMBER_ROLE, REMOVE_MEMBER } from "@/lib/models/members/gql"
import type {
    UpdateMemberRoleMutation,
    UpdateMemberRoleMutationVariables,
    RemoveMemberMutation,
    RemoveMemberMutationVariables
} from "@/lib/graphql/__generated__/graphql"

interface MembersDataTableProps {
    members: Member[]
    spaceId: string
    onMemberUpdated?: () => void
    isAdmin?: boolean
}

export function MembersDataTable({
    members,
    spaceId,
    onMemberUpdated,
    isAdmin = false
}: MembersDataTableProps) {
    const [selectedMember, setSelectedMember] = useState<Member | null>(null)
    const [pendingRole, setPendingRole] = useState<"admin" | "member" | null>(null)
    const [isUpdating, setIsUpdating] = useState(false)
    const [showConfirmDialog, setShowConfirmDialog] = useState(false)

    // Bulk selection state
    const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({})
    const [showBulkRoleDialog, setShowBulkRoleDialog] = useState(false)
    const [showBulkRemoveDialog, setShowBulkRemoveDialog] = useState(false)
    const [bulkRole, setBulkRole] = useState<"admin" | "member">("member")

    const [updateMemberRole] = useMutation<UpdateMemberRoleMutation, UpdateMemberRoleMutationVariables>(UPDATE_MEMBER_ROLE);
    const [removeMember] = useMutation<RemoveMemberMutation, RemoveMemberMutationVariables>(REMOVE_MEMBER);

    // Derive selected members from rowSelection
    const selectedMembers = Object.keys(rowSelection)
        .filter((k) => rowSelection[k])
        .map((k) => members[parseInt(k)])
        .filter(Boolean) as Member[]

    const handleRoleChange = (newRole: "admin" | "member") => {
        if (selectedMember && newRole !== selectedMember.role) {
            setPendingRole(newRole)
            setShowConfirmDialog(true)
        }
    }

    const confirmRoleChange = async () => {
        if (!selectedMember || !pendingRole) return

        setIsUpdating(true)
        try {
            await updateMemberRole({
                variables: {
                    spaceId,
                    userId: selectedMember.user.id,
                    role: pendingRole,
                }
            })

            // Update local state is tricky with useMutation unless we update cache manually.
            // But we call onMemberUpdated which refetches in parent.
            // For now, let's just rely on parent refetch.
            // We can optimistic update local state if we want, but generated types make it slightly complex if we construct objects incorrectly.
            // Actually, simply relying on parent refetch (onMemberUpdated) is safer and simpler for now.

            // setSelectedMember({ ...selectedMember, role: pendingRole }) // Can't easily construct full Member if type changed structure
            // Just close dialog and refresh
            onMemberUpdated?.()
        } catch (error) {
            console.error("Failed to update role:", error)
            alert("Failed to update role. You may not have permission.")
        } finally {
            setIsUpdating(false)
            setShowConfirmDialog(false)
            setPendingRole(null)
        }
    }

    const cancelRoleChange = () => {
        setShowConfirmDialog(false)
        setPendingRole(null)
    }

    // Bulk action handlers
    const handleBulkRoleChange = async () => {
        setIsUpdating(true)
        try {
            for (const member of selectedMembers) {
                if (member.role !== bulkRole) {
                    await updateMemberRole({
                        variables: {
                            spaceId,
                            userId: member.user.id,
                            role: bulkRole,
                        }
                    })
                }
            }
            onMemberUpdated?.()
            setRowSelection({})
        } catch (error) {
            console.error("Failed to update roles:", error)
            alert("Failed to update some roles. You may not have permission.")
        } finally {
            setIsUpdating(false)
            setShowBulkRoleDialog(false)
        }
    }

    const handleBulkRemove = async () => {
        setIsUpdating(true)
        try {
            for (const member of selectedMembers) {
                await removeMember({
                    variables: {
                        spaceId,
                        userId: member.user.id,
                    }
                })
            }
            onMemberUpdated?.()
            setRowSelection({})
        } catch (error) {
            console.error("Failed to remove members:", error)
            alert("Failed to remove some members. You may not have permission.")
        } finally {
            setIsUpdating(false)
            setShowBulkRemoveDialog(false)
        }
    }

    return (
        <>
            {/* Bulk Actions Toolbar */}
            {isAdmin && selectedMembers.length > 0 && (
                <div className="flex items-center gap-3 p-3 mb-4 bg-muted/50 border rounded-lg">
                    <div className="flex items-center gap-2 text-sm font-medium">
                        <UsersIcon className="h-4 w-4" />
                        {selectedMembers.length} selected
                    </div>
                    <Separator orientation="vertical" className="h-6" />
                    <div className="flex items-center gap-2">
                        <Select value={bulkRole} onValueChange={(v) => setBulkRole(v as "admin" | "member")}>
                            <SelectTrigger className="w-[130px] h-8">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="admin">
                                    <div className="flex items-center gap-2">
                                        <ShieldIcon className="h-3 w-3" />
                                        Admin
                                    </div>
                                </SelectItem>
                                <SelectItem value="member">
                                    <div className="flex items-center gap-2">
                                        <UserIcon className="h-3 w-3" />
                                        Member
                                    </div>
                                </SelectItem>
                            </SelectContent>
                        </Select>
                        <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => setShowBulkRoleDialog(true)}
                            disabled={isUpdating}
                        >
                            Change Role
                        </Button>
                        <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => setShowBulkRemoveDialog(true)}
                            disabled={isUpdating}
                        >
                            <TrashIcon className="h-4 w-4 mr-1" />
                            Remove
                        </Button>
                    </div>
                </div>
            )}

            <DataTable
                columns={columns}
                data={members}
                searchKey="user"
                searchPlaceholder="Search by name or email..."
                onRowClick={(member) => setSelectedMember(member)}
                rowSelection={rowSelection}
                onRowSelectionChange={setRowSelection}
            />

            {/* Member Details Dialog */}
            <Dialog open={!!selectedMember} onOpenChange={(open) => !open && setSelectedMember(null)}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle>Member Details</DialogTitle>
                        <DialogDescription>
                            {isAdmin ? "View and manage this member" : "Member information"}
                        </DialogDescription>
                    </DialogHeader>

                    {selectedMember && (
                        <div className="space-y-6">
                            {/* Profile Section */}
                            <div className="flex items-center gap-4">
                                <Avatar className="h-16 w-16">
                                    <AvatarFallback className="bg-primary/10 text-primary text-xl font-semibold">
                                        {selectedMember.user.name?.[0]?.toUpperCase()}
                                    </AvatarFallback>
                                </Avatar>
                                <div>
                                    <h3 className="text-xl font-semibold">
                                        {selectedMember.user.name}
                                    </h3>
                                    <div className="flex items-center gap-2 mt-1">
                                        {selectedMember.role === "admin" && (
                                            <Badge variant="default" className="gap-1">
                                                <ShieldIcon className="h-3 w-3" />
                                                Admin
                                            </Badge>
                                        )}
                                        {selectedMember.role === "member" && (
                                            <Badge variant="outline" className="gap-1">
                                                <UserIcon className="h-3 w-3" />
                                                Member
                                            </Badge>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <Separator />

                            {/* Details Grid */}
                            <div className="grid gap-4">
                                <div className="flex items-center gap-3">
                                    <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-muted">
                                        <MailIcon className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                    <div>
                                        <p className="text-xs text-muted-foreground">Email</p>
                                        <p className="text-sm font-medium">{selectedMember.user.email}</p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-muted">
                                        <CircleIcon className={`h-4 w-4 ${selectedMember.status === "active"
                                            ? "text-green-500 fill-green-500"
                                            : selectedMember.status === "pending"
                                                ? "text-yellow-500 fill-yellow-500"
                                                : "text-red-500 fill-red-500"
                                            }`} />
                                    </div>
                                    <div>
                                        <p className="text-xs text-muted-foreground">Status</p>
                                        <p className="text-sm font-medium capitalize">{selectedMember.status}</p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-muted">
                                        <CalendarIcon className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                    <div>
                                        <p className="text-xs text-muted-foreground">Joined</p>
                                        <p className="text-sm font-medium">
                                            {new Date(selectedMember.joinedAt as string).toLocaleDateString(undefined, {
                                                weekday: "long",
                                                year: "numeric",
                                                month: "long",
                                                day: "numeric",
                                            })}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-muted">
                                        <UserIcon className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                    <div>
                                        <p className="text-xs text-muted-foreground">User ID</p>
                                        <p className="text-sm font-mono text-muted-foreground">{selectedMember.user.id}</p>
                                    </div>
                                </div>
                            </div>

                            {/* Role Management (Admin Only) */}
                            {isAdmin && (
                                <>
                                    <Separator />
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold">Manage Role</h4>
                                        <div className="flex items-center gap-3">
                                            <Select
                                                value={selectedMember.role}
                                                onValueChange={(value) => handleRoleChange(value as "admin" | "member")}
                                                disabled={isUpdating}
                                            >
                                                <SelectTrigger className="w-[180px]">
                                                    <SelectValue />
                                                </SelectTrigger>
                                                <SelectContent>
                                                    <SelectItem value="admin">
                                                        <div className="flex items-center gap-2">
                                                            <ShieldIcon className="h-4 w-4" />
                                                            Admin
                                                        </div>
                                                    </SelectItem>
                                                    <SelectItem value="member">
                                                        <div className="flex items-center gap-2">
                                                            <UserIcon className="h-4 w-4" />
                                                            Member
                                                        </div>
                                                    </SelectItem>
                                                </SelectContent>
                                            </Select>
                                            {isUpdating && (
                                                <Loader2Icon className="h-4 w-4 animate-spin text-muted-foreground" />
                                            )}
                                        </div>
                                        <p className="text-xs text-muted-foreground">
                                            Admins can manage members, posts, and space settings.
                                        </p>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </DialogContent>
            </Dialog>

            {/* Confirmation Dialog */}
            <AlertDialog open={showConfirmDialog} onOpenChange={setShowConfirmDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Role Change</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to change {selectedMember?.user.name}&apos;s role from{" "}
                            <strong className="text-foreground">{selectedMember?.role}</strong> to{" "}
                            <strong className="text-foreground">{pendingRole}</strong>?
                            {pendingRole === "admin" && (
                                <span className="block mt-2 text-yellow-600">
                                    ⚠️ Admins have full control over the space, including managing other members.
                                </span>
                            )}
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel onClick={cancelRoleChange} disabled={isUpdating}>
                            Cancel
                        </AlertDialogCancel>
                        <AlertDialogAction onClick={confirmRoleChange} disabled={isUpdating}>
                            {isUpdating ? (
                                <>
                                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                                    Updating...
                                </>
                            ) : (
                                "Confirm"
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Bulk Role Change Confirmation */}
            <AlertDialog open={showBulkRoleDialog} onOpenChange={setShowBulkRoleDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Bulk Role Change</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to change the role of <strong className="text-foreground">{selectedMembers.length} members</strong> to{" "}
                            <strong className="text-foreground">{bulkRole}</strong>?
                            {bulkRole === "admin" && (
                                <span className="block mt-2 text-yellow-600">
                                    ⚠️ Admins have full control over the space, including managing other members.
                                </span>
                            )}
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={isUpdating}>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleBulkRoleChange} disabled={isUpdating}>
                            {isUpdating ? (
                                <>
                                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                                    Updating...
                                </>
                            ) : (
                                "Confirm"
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Bulk Remove Confirmation */}
            <AlertDialog open={showBulkRemoveDialog} onOpenChange={setShowBulkRemoveDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm Bulk Remove</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to remove <strong className="text-foreground">{selectedMembers.length} members</strong> from this space?
                            <span className="block mt-2 text-destructive">
                                ⚠️ This action cannot be undone. Members will need to rejoin the space.
                            </span>
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={isUpdating}>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleBulkRemove}
                            disabled={isUpdating}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            {isUpdating ? (
                                <>
                                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                                    Removing...
                                </>
                            ) : (
                                "Remove Members"
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </>
    )
}
