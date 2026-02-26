import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { sql, eq } from "drizzle-orm";

async function main() {
    // Check current user (andreazago)
    const me = await db.query.users.findFirst({
        where: eq(users.username, "andreazago"),
    });

    if (!me) { console.log("User not found"); process.exit(1); }

    console.log(`\nCurrent user: ${me.username}`);
    console.log(`Coordinates: ${JSON.stringify(me.coordinates)}`);
    console.log(`Location: ${me.location}`);

    if (!me.coordinates) { console.log("NO COORDS → findMatches returns []"); process.exit(0); }

    const myCoords = me.coordinates as { x: number; y: number };

    // Simulate the findMatches query — check candidates within 50km
    const candidates = await db
        .select({
            username: users.username,
            name: users.name,
            location: users.location,
            distanceKm: sql<number>`ST_DistanceSphere(${users.coordinates}, ST_GeomFromText(${`POINT(${myCoords.x} ${myCoords.y})`}, 4326)) / 1000`,
        })
        .from(users)
        .where(
            sql`${users.id} != ${me.id}
        AND ${users.coordinates} IS NOT NULL
        AND ${users.username} IS NOT NULL
        AND ${users.birthdate} IS NOT NULL
        AND ST_DistanceSphere(${users.coordinates}, ST_GeomFromText(${`POINT(${myCoords.x} ${myCoords.y})`}, 4326)) <= ${50 * 1000}`
        )
        .limit(20);

    console.log(`\nCandidates within 50km: ${candidates.length}`);
    for (const c of candidates) {
        console.log(`  ${String(c.username).padEnd(22)} km=${c.distanceKm?.toFixed(1).padStart(6)} location=${c.location}`);
    }

    process.exit(0);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
