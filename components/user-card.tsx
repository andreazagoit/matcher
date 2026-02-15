import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";

interface UserCardUser {
    id?: string;
    name?: string;
    givenName?: string;
    familyName?: string;
    email?: string;
    image?: string | null;
    gender?: string | null;
    birthdate?: string;
}

interface UserCardProps {
    user: UserCardUser;
    compatibility?: number;
}

export function UserCard({ user, compatibility }: UserCardProps) {
    const name = user.name
        || `${user.givenName ?? ""} ${user.familyName ?? ""}`.trim()
        || user.email
        || "Unknown User";
    const initials = (user.givenName?.[0] || "") + (user.familyName?.[0] || "");

    return (
        <Card className="overflow-hidden hover:shadow-md transition-shadow duration-200">
            <CardHeader className="p-0">
                <div className="relative aspect-square w-full">
                    <Avatar className="w-full h-full rounded-none">
                        <AvatarImage
                            src={user.image ?? ""}
                            alt={name}
                            className="object-cover"
                        />
                        <AvatarFallback className="rounded-none text-2xl">
                            {initials || "?"}
                        </AvatarFallback>
                    </Avatar>
                    {compatibility !== undefined && (
                        <div className="absolute top-2 right-2">
                            <Badge variant="secondary" className="bg-background/80 backdrop-blur-sm font-bold">
                                {Math.round(compatibility * 100)}% Match
                            </Badge>
                        </div>
                    )}
                </div>
            </CardHeader>
            <CardContent className="p-4">
                <div className="space-y-1">
                    <h3 className="font-semibold text-lg leading-none truncate">{name}</h3>
                    {user.email && (
                        <p className="text-sm text-muted-foreground truncate">{user.email}</p>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
