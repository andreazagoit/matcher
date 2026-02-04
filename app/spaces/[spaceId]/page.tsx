"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CopyIcon, UsersIcon, RefreshCwIcon, TrashIcon, EllipsisVerticalIcon, ShieldIcon, ShieldCheckIcon } from "lucide-react";
import { graphql } from "@/lib/graphql/client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  clientId: string;
  // stored locally in session storage only on creation, 
  // but we might want a way to rotate/view it if we keep that logic
  // For now, let's assume we can't view it again unless rotated
  isActive: boolean;
  isPublic: boolean;
  requiresApproval: boolean;
  membersCount: number;
  createdAt: string;
  owner: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
  };
  members?: Member[];
}

interface Member {
  id: string;
  role: string;
  status: string;
  joinedAt: string;
  user: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
  };
}

export default function SpaceDetailPage() {
  const params = useParams();
  const router = useRouter();
  const spaceId = params.spaceId as string;

  const [space, setSpace] = useState<Space | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchSpace = async () => {
    try {
      const data = await graphql<{ space: Space }>(`
        query GetSpace($id: ID!) {
          space(id: $id) {
            id
            name
            slug
            description
            clientId
            isActive
            isPublic
            requiresApproval
            membersCount
            createdAt
            owner {
              id
              firstName
              lastName
              email
            }
            members(limit: 10) {
              id
              role
              status
              joinedAt
              user {
                id
                firstName
                lastName
                email
              }
            }
          }
        }
      `, { id: spaceId });

      if (data.space) {
        setSpace(data.space);
      } else {
        router.push("/spaces");
      }
    } catch (error) {
      console.error("Failed to fetch space:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSpace();
  }, [spaceId]);

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this space? This cannot be undone.")) return;

    try {
      await graphql(`
        mutation DeleteSpace($id: ID!) {
          deleteSpace(id: $id)
        }
      `, { id: spaceId });

      router.push("/spaces");
    } catch (error) {
      console.error("Failed to delete space", error);
      alert("Failed to delete space");
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <RefreshCwIcon className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!space) return null;

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-6">
        <Link href="/spaces" className="hover:text-foreground transition">Spaces</Link>
        <span>/</span>
        <span className="text-foreground">{space.name}</span>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-xl bg-primary/10 flex items-center justify-center text-primary text-2xl font-bold">
            {space.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
              {space.name}
              <Badge variant={space.isPublic ? "outline" : "secondary"}>
                {space.isPublic ? "Public" : "Private"}
              </Badge>
            </h1>
            <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
              <span className="font-mono bg-muted px-2 py-0.5 rounded text-xs">{space.slug}</span>
              <span>•</span>
              <span>{space.membersCount} members</span>
            </div>
            {space.description && (
              <p className="text-muted-foreground mt-2">{space.description}</p>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Link href={`/spaces/${spaceId}/settings`}>
            <Button variant="outline">Settings</Button>
          </Link>
          <Button variant="destructive" size="icon" onClick={handleDelete}>
            <TrashIcon className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="members" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-[400px]">
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="developers">Developers</TabsTrigger>
        </TabsList>

        <TabsContent value="members" className="mt-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Members</CardTitle>
                  <CardDescription>Manage community members</CardDescription>
                </div>
                {/* Add Invite Button here later */}
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>User</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Joined</TableHead>
                    <TableHead className="text-right"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {space.members?.map((member) => (
                    <TableRow key={member.id}>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          <Avatar className="h-8 w-8">
                            <AvatarFallback>{member.user.firstName[0]}{member.user.lastName[0]}</AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-medium text-sm">{member.user.firstName} {member.user.lastName}</p>
                            <p className="text-xs text-muted-foreground">{member.user.email}</p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        {member.role === 'owner' && <Badge variant="default" className="gap-1"><ShieldCheckIcon className="h-3 w-3" /> Owner</Badge>}
                        {member.role === 'admin' && <Badge variant="secondary" className="gap-1"><ShieldIcon className="h-3 w-3" /> Admin</Badge>}
                        {member.role === 'member' && <Badge variant="outline">Member</Badge>}
                      </TableCell>
                      <TableCell>
                        <span className={`capitalize text-sm ${member.status === 'active' ? 'text-green-600' : 'text-yellow-600'}`}>
                          {member.status}
                        </span>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {new Date(member.joinedAt).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon-xs">
                              <EllipsisVerticalIcon className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Actions</DropdownMenuLabel>
                            <DropdownMenuItem>Promote to Admin</DropdownMenuItem>
                            <DropdownMenuItem className="text-destructive">Remove</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="developers" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Developer Credentials</CardTitle>
              <CardDescription>Use these to integrate your custom apps with this Space</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Client ID */}
              <div className="space-y-2">
                <Label>Client ID</Label>
                <InputGroup>
                  <InputGroupInput
                    value={space.clientId}
                    readOnly
                    className="font-mono bg-muted"
                  />
                  <InputGroupAddon align="inline-end">
                    <InputGroupButton
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => copyToClipboard(space.clientId)}
                    >
                      <CopyIcon className="h-4 w-4" />
                    </InputGroupButton>
                  </InputGroupAddon>
                </InputGroup>
              </div>

              <div className="bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 p-4 rounded-lg text-sm">
                Secret keys are only shown once upon creation or rotation. If you lost your secret key, you can generate a new one in Settings.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
