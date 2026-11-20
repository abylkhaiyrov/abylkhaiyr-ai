import { createContext, useContext } from "react";
import type { Role, User } from "@/lib/api";

export const UserContext = createContext<User | null>(null);

export function useCurrentUser(): User {
  const user = useContext(UserContext);
  if (!user) throw new Error("UserContext is missing");
  return user;
}

export function useRole(): Role {
  return useCurrentUser().role;
}
