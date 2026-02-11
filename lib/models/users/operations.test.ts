import { describe, it, expect, beforeAll } from "vitest";
import { getAllUsers } from "./operations";
import { findMatches } from "@/lib/models/profiles/operations";

describe("findMatches", () => {
  let testUserId: string;

  beforeAll(async () => {
    // Retrieve the first available user for testing
    const users = await getAllUsers();
    if (users.length === 0) {
      throw new Error("No users found. Run db:seed first.");
    }
    testUserId = users[0].id;
  });

  it("should find matches for an existing user", async () => {
    const matches = await findMatches(testUserId, { limit: 5 });

    expect(matches).toBeDefined();
    expect(Array.isArray(matches)).toBe(true);
    expect(matches.length).toBeLessThanOrEqual(5);
  });

  it("should return matches with a similarity score", async () => {
    const matches = await findMatches(testUserId, { limit: 3 });

    if (matches.length > 0) {
      const match = matches[0];
      expect(match).toHaveProperty("user");
      expect(match).toHaveProperty("profile");
      expect(match).toHaveProperty("similarity");
      expect(match).toHaveProperty("breakdown");
      expect(typeof match.similarity).toBe("number");

      // Check user fields
      expect(match.user).toHaveProperty("id");
      expect(match.user).toHaveProperty("firstName");
      expect(match.user).toHaveProperty("lastName");
    }
  });

  it("should exclude the current user from the results", async () => {
    const matches = await findMatches(testUserId, { limit: 10 });

    const userIds = matches.map((m) => m.user.id);
    expect(userIds).not.toContain(testUserId);
  });

  it("should sort matches by descending similarity", async () => {
    const matches = await findMatches(testUserId, { limit: 10 });

    if (matches.length > 1) {
      for (let i = 0; i < matches.length - 1; i++) {
        expect(matches[i].similarity).toBeGreaterThanOrEqual(
          matches[i + 1].similarity
        );
      }
    }
  });

  it("should respect the limit parameter", async () => {
    const matches3 = await findMatches(testUserId, { limit: 3 });
    const matches5 = await findMatches(testUserId, { limit: 5 });

    expect(matches3.length).toBeLessThanOrEqual(3);
    expect(matches5.length).toBeLessThanOrEqual(5);
  });

  it("should throw an error for a non-existent user", async () => {
    const fakeUserId = "00000000-0000-0000-0000-000000000000";

    await expect(findMatches(fakeUserId)).rejects.toThrow("Profile not found");
  });

  it("should accept custom matching weights", async () => {
    const matchesDefault = await findMatches(testUserId, { limit: 5 });
    const matchesWeights = await findMatches(testUserId, {
      limit: 5,
      weights: {
        psychological: 0.8,
        values: 0.1,
        interests: 0.05,
        behavioral: 0.05,
      },
    });

    // Entrambi dovrebbero funzionare
    expect(matchesDefault).toBeDefined();
    expect(matchesWeights).toBeDefined();
  });
});

