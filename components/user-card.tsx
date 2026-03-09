import Link from "next/link";

interface UserCardUser {
  id: string;
  username?: string | null;
  name: string;
  image?: string | null;
  birthdate: string;
  gender?: string | null;
  userItems?: { id: string; type: string; content: string; displayOrder: number }[] | null;
}

interface UserCardProps {
  user: UserCardUser;
  compatibility?: number;
}

function getAge(birthdate: string): number | null {
  const date = new Date(birthdate);
  if (Number.isNaN(date.getTime())) return null;
  const now = new Date();
  let age = now.getFullYear() - date.getFullYear();
  if (
    now.getMonth() < date.getMonth() ||
    (now.getMonth() === date.getMonth() && now.getDate() < date.getDate())
  ) age -= 1;
  return age >= 0 ? age : null;
}

export function UserCard({ user, compatibility }: UserCardProps) {
  const age = getAge(user.birthdate);
  const initials = user.name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const photo =
    user.userItems
      ?.filter((i) => i.type === "photo")
      .sort((a, b) => a.displayOrder - b.displayOrder)[0]?.content ??
    user.image ??
    null;

  const inner = (
    <div className="relative aspect-[4/5] w-full overflow-hidden rounded-2xl bg-muted group cursor-pointer">
      {photo ? (
        <img
          src={photo}
          alt={user.name}
          className="absolute inset-0 size-full object-cover transition-transform duration-500 group-hover:scale-[1.04]"
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted to-muted-foreground/20">
          <span className="text-5xl font-black text-muted-foreground/30 select-none">
            {initials || "?"}
          </span>
        </div>
      )}

      {/* Graduated blur — stacked layers with increasing blur and decreasing height */}
      <div className="absolute inset-x-0 bottom-0 h-32 backdrop-blur-[8px]  [mask-image:linear-gradient(to_top,black_0%,black_20%,transparent_50%)]" />
      <div className="absolute inset-x-0 bottom-0 h-32 backdrop-blur-[4px]  [mask-image:linear-gradient(to_top,transparent_20%,black_40%,transparent_65%)]" />
      <div className="absolute inset-x-0 bottom-0 h-32 backdrop-blur-[2px]  [mask-image:linear-gradient(to_top,transparent_40%,black_60%,transparent_80%)]" />

      {/* Dark scrim on top of blur for text legibility */}
      <div className="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-black/65 via-black/25 to-transparent" />

      {/* Text content */}
      <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 px-3.5 pb-3.5">
        <div className="min-w-0">
          <p className="truncate text-[15px] font-semibold leading-snug text-white">
            {user.name}
          </p>
          {age !== null && (
            <p className="text-[13px] font-normal text-white/80 leading-none mt-0.5">
              {age} anni
            </p>
          )}
        </div>

        {compatibility !== undefined && (
          <span className="shrink-0 rounded-full bg-white/15 px-2.5 py-1 text-xs font-bold text-white backdrop-blur-sm ring-1 ring-white/25 mb-0.5">
            {Math.round(compatibility * 100)}%
          </span>
        )}
      </div>
    </div>
  );

  if (!user.username) return inner;
  return (
    <Link href={`/users/${user.username}`} className="block">
      {inner}
    </Link>
  );
}
