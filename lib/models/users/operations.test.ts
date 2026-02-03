import { describe, it, expect, beforeAll } from "vitest";
import { findMatches, getAllUsers } from "./operations";

describe("findMatches", () => {
  let testUserId: string;

  beforeAll(async () => {
    // Ottieni il primo utente per i test
    const users = await getAllUsers();
    if (users.length === 0) {
      throw new Error("No users found. Run db:seed first.");
    }
    testUserId = users[0].id;
  });

  it("dovrebbe trovare match per un utente esistente", async () => {
    const matches = await findMatches(testUserId, { limit: 5 });

    expect(matches).toBeDefined();
    expect(Array.isArray(matches)).toBe(true);
    expect(matches.length).toBeLessThanOrEqual(5);
  });

  it("dovrebbe restituire match con similarity score", async () => {
    const matches = await findMatches(testUserId, { limit: 3 });

    if (matches.length > 0) {
      const match = matches[0];
      expect(match).toHaveProperty("id");
      expect(match).toHaveProperty("firstName");
      expect(match).toHaveProperty("lastName");
      expect(match).toHaveProperty("email");
      expect(match).toHaveProperty("values");
      expect(match).toHaveProperty("interests");
      expect(match).toHaveProperty("similarity");
      expect(typeof match.similarity).toBe("number");
    }
  });

  it("dovrebbe escludere l'utente corrente dai risultati", async () => {
    const matches = await findMatches(testUserId, { limit: 10 });

    const userIds = matches.map((m) => m.id);
    expect(userIds).not.toContain(testUserId);
  });

  it("dovrebbe ordinare i match per similarity decrescente", async () => {
    const matches = await findMatches(testUserId, { limit: 10 });

    if (matches.length > 1) {
      for (let i = 0; i < matches.length - 1; i++) {
        expect(matches[i].similarity).toBeGreaterThanOrEqual(
          matches[i + 1].similarity
        );
      }
    }
  });

  it("dovrebbe rispettare il parametro limit", async () => {
    const matches3 = await findMatches(testUserId, { limit: 3 });
    const matches5 = await findMatches(testUserId, { limit: 5 });

    expect(matches3.length).toBeLessThanOrEqual(3);
    expect(matches5.length).toBeLessThanOrEqual(5);
  });

  it("dovrebbe lanciare errore per utente non esistente", async () => {
    const fakeUserId = "00000000-0000-0000-0000-000000000000";

    await expect(findMatches(fakeUserId)).rejects.toThrow("User not found");
  });

  it("dovrebbe accettare pesi personalizzati", async () => {
    const matchesDefault = await findMatches(testUserId, { limit: 5 });
    const matchesValuesHeavy = await findMatches(testUserId, {
      limit: 5,
      valuesWeight: 0.8,
      interestsWeight: 0.2,
    });

    // Entrambi dovrebbero funzionare
    expect(matchesDefault).toBeDefined();
    expect(matchesValuesHeavy).toBeDefined();
  });
});

