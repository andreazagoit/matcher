import { db } from "@/lib/db/drizzle";
import { users } from "./schema";
import type { User, NewUser } from "./schema";
import {
  createUserSchema,
  updateUserSchema,
  type CreateUserInput,
  type UpdateUserInput,
} from "./validator";
import { userItems } from "@/lib/models/useritems/schema";
import { eq, and, gte, inArray, sql } from "drizzle-orm";
import { GraphQLError } from "graphql";
import { embeddings } from "@/lib/models/embeddings/schema";
import { events } from "@/lib/models/events/schema";
import { eventAttendees } from "@/lib/models/events/schema";
import { embedUser } from "@/lib/ml/client";

/**
 * Create a new user record, optionally with profile photos and prompts.
 * All inserts happen in a single transaction.
 */
export async function createUser(
  input: CreateUserInput
): Promise<User> {
  // Sanitize username
  if (input.username) {
    input.username = input.username.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "").slice(0, 30);
  }

  const validatedInput = createUserSchema.parse(input);

  const existing = await db.query.users.findFirst({
    where: eq(users.username, validatedInput.username),
  });
  if (existing) {
    throw new GraphQLError("Username already taken", { extensions: { code: "USERNAME_TAKEN" } });
  }

  const newUser = await db.transaction(async (tx) => {
    const [created] = await tx
      .insert(users)
      .values({
        username: validatedInput.username,
        name: validatedInput.name,
        email: validatedInput.email,
        birthdate: validatedInput.birthdate,
        ...(validatedInput.gender && { gender: validatedInput.gender }),
        ...(validatedInput.sexualOrientation && { sexualOrientation: validatedInput.sexualOrientation }),
        ...(validatedInput.heightCm !== undefined && { heightCm: validatedInput.heightCm }),
        ...(validatedInput.relationshipIntent && { relationshipIntent: validatedInput.relationshipIntent }),
        ...(validatedInput.relationshipStyle && { relationshipStyle: validatedInput.relationshipStyle }),
        ...(validatedInput.hasChildren && { hasChildren: validatedInput.hasChildren }),
        ...(validatedInput.wantsChildren && { wantsChildren: validatedInput.wantsChildren }),
        ...(validatedInput.religion && { religion: validatedInput.religion }),
        ...(validatedInput.smoking && { smoking: validatedInput.smoking }),
        ...(validatedInput.drinking && { drinking: validatedInput.drinking }),
        ...(validatedInput.activityLevel && { activityLevel: validatedInput.activityLevel }),
        ...(validatedInput.jobTitle && { jobTitle: validatedInput.jobTitle }),
        ...(validatedInput.educationLevel && { educationLevel: validatedInput.educationLevel }),
        ...(validatedInput.schoolName && { schoolName: validatedInput.schoolName }),
        ...(validatedInput.languages && { languages: validatedInput.languages }),
        ...(validatedInput.ethnicity && { ethnicity: validatedInput.ethnicity }),
        ...(validatedInput.location && { location: validatedInput.location }),
      })
      .returning();

    // Insert photos and prompts if provided
    const allItems = [
      ...(validatedInput.photos ?? []).map((p, i) => ({
        userId: created.id,
        type: "photo" as const,
        promptKey: null,
        content: p.content,
        displayOrder: i,
      })),
      ...(validatedInput.prompts ?? []).map((p, i) => ({
        userId: created.id,
        type: "prompt" as const,
        promptKey: p.promptKey,
        content: p.content,
        displayOrder: (validatedInput.photos?.length ?? 0) + i,
      })),
    ];

    if (allItems.length > 0) {
      await tx.insert(userItems).values(allItems);
      // Sync users.image to first photo
      const firstPhoto = allItems.find((i) => i.type === "photo");
      if (firstPhoto) {
        await tx.update(users).set({ image: firstPhoto.content }).where(eq(users.id, created.id));
        created.image = firstPhoto.content;
      }
    }

    return created;
  });

  // Generate and store embedding
  const embedding = await embedUser({
    birthdate: validatedInput.birthdate,
    gender: validatedInput.gender,
    relationshipIntent: validatedInput.relationshipIntent,
    smoking: validatedInput.smoking,
    drinking: validatedInput.drinking,
    activityLevel: validatedInput.activityLevel,
  });
  if (embedding) {
    await db
      .insert(embeddings)
      .values({ entityId: newUser.id, entityType: "user", embedding })
      .onConflictDoUpdate({
        target: [embeddings.entityId, embeddings.entityType],
        set: { embedding, updatedAt: new Date() },
      });
  }

  return newUser;
}

/**
 * Update an existing user record with partial data.
 * If photos or prompts are provided, they fully replace the existing items of that type.
 */
export async function updateUser(
  id: string,
  input: UpdateUserInput
): Promise<User> {
  const validatedInput = updateUserSchema.parse(input);

  const existingUser = await db.query.users.findFirst({
    where: eq(users.id, id),
  });
  if (!existingUser) {
    throw new Error("User not found");
  }

  const updateData: Partial<NewUser> = { updatedAt: new Date() };

  if (validatedInput.username !== undefined) {
    const taken = await db.query.users.findFirst({ where: eq(users.username, validatedInput.username) });
    if (taken && taken.id !== id) {
      throw new GraphQLError("Username already taken", { extensions: { code: "USERNAME_TAKEN" } });
    }
    updateData.username = validatedInput.username;
  }
  if (validatedInput.name !== undefined) updateData.name = validatedInput.name;
  if (validatedInput.email !== undefined) updateData.email = validatedInput.email;
  if (validatedInput.birthdate !== undefined) updateData.birthdate = validatedInput.birthdate;
  if (validatedInput.gender !== undefined) updateData.gender = validatedInput.gender;
  if (validatedInput.sexualOrientation !== undefined) updateData.sexualOrientation = validatedInput.sexualOrientation;
  if (validatedInput.heightCm !== undefined) updateData.heightCm = validatedInput.heightCm;
  if (validatedInput.relationshipIntent !== undefined) updateData.relationshipIntent = validatedInput.relationshipIntent;
  if (validatedInput.relationshipStyle !== undefined) updateData.relationshipStyle = validatedInput.relationshipStyle;
  if (validatedInput.hasChildren !== undefined) updateData.hasChildren = validatedInput.hasChildren;
  if (validatedInput.wantsChildren !== undefined) updateData.wantsChildren = validatedInput.wantsChildren;
  if (validatedInput.religion !== undefined) updateData.religion = validatedInput.religion;
  if (validatedInput.smoking !== undefined) updateData.smoking = validatedInput.smoking;
  if (validatedInput.drinking !== undefined) updateData.drinking = validatedInput.drinking;
  if (validatedInput.activityLevel !== undefined) updateData.activityLevel = validatedInput.activityLevel;
  if (validatedInput.jobTitle !== undefined) updateData.jobTitle = validatedInput.jobTitle;
  if (validatedInput.educationLevel !== undefined) updateData.educationLevel = validatedInput.educationLevel;
  if (validatedInput.schoolName !== undefined) updateData.schoolName = validatedInput.schoolName;
  if (validatedInput.languages !== undefined) updateData.languages = validatedInput.languages;
  if (validatedInput.ethnicity !== undefined) updateData.ethnicity = validatedInput.ethnicity;
  if (validatedInput.location !== undefined) updateData.location = validatedInput.location;

  const updatedUser = await db.transaction(async (tx) => {
    const [updated] = await tx
      .update(users)
      .set(updateData)
      .where(eq(users.id, id))
      .returning();

    // Replace photos if provided
    if (validatedInput.photos !== undefined) {
      await tx.delete(userItems).where(
        and(eq(userItems.userId, id), eq(userItems.type, "photo"))
      );
      if (validatedInput.photos.length > 0) {
        await tx.insert(userItems).values(
          validatedInput.photos.map((p, i) => ({
            userId: id,
            type: "photo" as const,
            promptKey: null,
            content: p.content,
            displayOrder: i,
          }))
        );
        // Sync users.image to first photo
        await tx.update(users)
          .set({ image: validatedInput.photos[0].content })
          .where(eq(users.id, id));
        updated.image = validatedInput.photos[0].content;
      } else {
        await tx.update(users).set({ image: null }).where(eq(users.id, id));
        updated.image = null;
      }
    }

    // Replace prompts if provided
    if (validatedInput.prompts !== undefined) {
      await tx.delete(userItems).where(
        and(eq(userItems.userId, id), eq(userItems.type, "prompt"))
      );
      if (validatedInput.prompts.length > 0) {
        // displayOrder continues after photos
        const photoCount = validatedInput.photos?.length
          ?? (await tx.select().from(userItems).where(
            and(eq(userItems.userId, id), eq(userItems.type, "photo"))
          )).length;
        await tx.insert(userItems).values(
          validatedInput.prompts.map((p, i) => ({
            userId: id,
            type: "prompt" as const,
            promptKey: p.promptKey,
            content: p.content,
            displayOrder: photoCount + i,
          }))
        );
      }
    }

    return updated;
  });

  // Regenerate embedding if any profile-relevant field changed
  const profileFields = ["gender", "relationshipIntent", "smoking", "drinking", "activityLevel", "birthdate"] as const;
  const hasProfileChange = profileFields.some((f) => validatedInput[f] !== undefined);
  if (hasProfileChange) {
    const merged = { ...existingUser, ...updatedUser };
    const embedding = await embedUser({
      birthdate: merged.birthdate,
      gender: merged.gender,
      relationshipIntent: merged.relationshipIntent ?? [],
      smoking: merged.smoking,
      drinking: merged.drinking,
      activityLevel: merged.activityLevel,
    });
    if (embedding) {
      await db
        .insert(embeddings)
        .values({ entityId: id, entityType: "user", embedding })
        .onConflictDoUpdate({
          target: [embeddings.entityId, embeddings.entityType],
          set: { embedding, updatedAt: new Date() },
        });
    }
  }

  return updatedUser;
}

/**
 * Retrieve a user by their unique ID.
 */
export async function getUserById(id: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.id, id),
  });
  return result || null;
}

/**
 * Retrieve a user by their username.
 */
export async function getUserByUsername(username: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.username, username),
  });
  return result || null;
}

/**
 * Retrieve a user by their email address.
 */
export async function getUserByEmail(email: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.email, email),
  });
  return result || null;
}

/**
 * Retrieve all user records from the database.
 */
export async function getAllUsers(): Promise<User[]> {
  return await db.query.users.findMany();
}

/**
 * Permanently delete a user record by ID.
 */
export async function deleteUser(id: string): Promise<boolean> {
  const [deleted] = await db
    .delete(users)
    .where(eq(users.id, id))
    .returning();
  return !!deleted;
}

/**
 * AI-recommended events for a user — based on cosine similarity between the
 * user's embedding and each event's embedding. Excludes already-attended events.
 */
export async function getUserRecommendedEvents(
  userId: string,
  limit = 10,
  offset = 0,
) {
  const row = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
    columns: { embedding: true },
  });
  if (!row) return [];

  const attended = await db
    .select({ eventId: eventAttendees.eventId })
    .from(eventAttendees)
    .where(eq(eventAttendees.userId, userId));
  const excludeIds = attended.map((a) => a.eventId);

  const vec = `[${row.embedding.join(",")}]`;
  const rows = await db.execute<{ entity_id: string }>(sql`
    SELECT entity_id FROM embeddings
    WHERE  entity_type = 'event'
    ${excludeIds.length ? sql`AND entity_id != ALL(${excludeIds})` : sql``}
    ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
    LIMIT ${sql.raw(String(limit))} OFFSET ${sql.raw(String(offset))}
  `);

  const ids = rows.map((r) => r.entity_id);
  if (!ids.length) return [];

  const result = await db
    .select()
    .from(events)
    .where(and(inArray(events.id, ids), gte(events.startsAt, new Date())));
  const map = new Map(result.map((e) => [e.id, e]));
  return ids.map((id) => map.get(id)).filter(Boolean);
}

export async function isUsernameTaken(username: string): Promise<boolean> {
  const existing = await db.query.users.findFirst({
    where: eq(users.username, username),
    columns: { id: true },
  });
  return !!existing;
}

/**
 * Update a user's location (PostGIS point).
 * PostGIS convention: x = longitude, y = latitude.
 * If locationOverride is provided, it is used directly instead of reverse geocoding.
 */
export async function updateUserLocation(
  id: string,
  lat: number,
  lon: number,
  locationOverride?: string,
): Promise<User> {
  const updateData: Partial<User> = {
    coordinates: { x: lon, y: lat },
    locationUpdatedAt: new Date(),
    updatedAt: new Date(),
  };

  if (locationOverride) {
    updateData.location = locationOverride;
  } else {
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`, {
        headers: { "User-Agent": "MatcherApp/1.0" }
      });
      if (res.ok) {
        const data = await res.json();
        const city = data.address?.city || data.address?.town || data.address?.village || data.name;
        if (city) updateData.location = city;
      }
    } catch (err) {
      console.error("Reverse geocoding error:", err);
    }
  }

  const [updated] = await db
    .update(users)
    .set(updateData)
    .where(eq(users.id, id))
    .returning();

  if (!updated) throw new Error("User not found");
  return updated;
}

