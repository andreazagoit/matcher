/**
 * User Account API
 * GET /api/auth/account - Get current user details
 * PUT /api/auth/account - Update current user details
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { getUserById, updateUser } from "@/lib/models/users/operations";
import { updateUserSchema } from "@/lib/models/users/validator";
import { z } from "zod";

export async function GET() {
    try {
        const cookieStore = await cookies();
        const userId = cookieStore.get("user_id")?.value;

        if (!userId) {
            return Response.json({ error: "Unauthorized" }, { status: 401 });
        }

        const user = await getUserById(userId);
        if (!user) {
            return Response.json({ error: "User not found" }, { status: 404 });
        }

        return Response.json({
            user: {
                id: user.id,
                firstName: user.firstName,
                lastName: user.lastName,
                email: user.email,
                birthDate: user.birthDate,
                gender: user.gender,
            }
        });
    } catch (error) {
        console.error("Get account error:", error);
        return Response.json(
            { error: "Failed to fetch account details" },
            { status: 500 }
        );
    }
}

export async function PUT(request: NextRequest) {
    try {
        const cookieStore = await cookies();
        const userId = cookieStore.get("user_id")?.value;

        if (!userId) {
            return Response.json({ error: "Unauthorized" }, { status: 401 });
        }

        const body = await request.json();
        const validatedData = updateUserSchema.parse(body);

        const updatedUser = await updateUser(userId, validatedData);

        return Response.json({
            user: {
                id: updatedUser.id,
                firstName: updatedUser.firstName,
                lastName: updatedUser.lastName,
                email: updatedUser.email,
                birthDate: updatedUser.birthDate,
                gender: updatedUser.gender,
            }
        });
    } catch (error) {
        console.error("Update account error:", error);
        if (error instanceof z.ZodError) {
            return Response.json({ error: error.issues }, { status: 400 });
        }
        return Response.json(
            { error: "Failed to update account" },
            { status: 500 }
        );
    }
}
