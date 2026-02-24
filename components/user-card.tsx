import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
interface UserCardProps {
    user: {
        id: string;
        username?: string | null;
        name: string;
        image?: string | null;
        birthdate: string;
        gender?: string | null;
        userItems?: { id: string; type: string; content: string; displayOrder: number }[] | null;
    };
    compatibility?: number;
}

function getAge(birthdate: string): number | null {
    const date = new Date(birthdate);
    if (Number.isNaN(date.getTime())) return null;

    const now = new Date();
    let age = now.getFullYear() - date.getFullYear();
    const monthDiff = now.getMonth() - date.getMonth();
    const dayDiff = now.getDate() - date.getDate();

    if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
        age -= 1;
    }
    return age >= 0 ? age : null;
}

export function UserCard({ user, compatibility }: UserCardProps) {
    const name = user.name;
    const initials = name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
    const age = getAge(user.birthdate);
    const firstPhoto = user.userItems?.find((item) => item.type === "photo")?.content ?? user.image ?? "";
    const canOpenProfile = !!user.username;
    const card = (
        <Card className="overflow-hidden hover:shadow-md transition-shadow duration-200">
            <CardHeader className="p-4">
                <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-3">
                        <Avatar className="h-10 w-10 shrink-0">
                            <AvatarImage src={user.image ?? ""} alt={name} />
                            <AvatarFallback>{initials || "?"}</AvatarFallback>
                        </Avatar>
                        <div className="min-w-0">
                            <h3 className="truncate text-base font-semibold leading-none">
                                {name}
                                {age !== null ? `, ${age}` : ""}
                            </h3>
                        </div>
                    </div>
                    {compatibility !== undefined && (
                        <Badge variant="secondary" className="font-bold">
                            {Math.round(compatibility * 100)}% Match
                        </Badge>
                    )}
                </div>
            </CardHeader>
            <CardContent className="p-0">
                <div className="relative aspect-square w-full bg-muted/30">
                    <Avatar className="h-full w-full rounded-none">
                        <AvatarImage
                            src={firstPhoto}
                            alt={name}
                            className="object-cover"
                        />
                        <AvatarFallback className="rounded-none text-2xl">
                            {initials || "?"}
                        </AvatarFallback>
                    </Avatar>
                </div>
            </CardContent>
        </Card>
    );

    if (!canOpenProfile) return card;
    return (
        <Link href={`/users/${user.username}`} className="block">
            {card}
        </Link>
    );
}
